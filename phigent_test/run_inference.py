"""
在 phigent pkl 数据上跑 SparseDriveV2 推理。

数据来源:
  manifest pkl  -> list[{iid, extra_annotation_path}]
  iid pkl       -> 单帧 dict,关键字段:
                   cams[cam1/cam4/cam8/...] = {
                     'data_path': 图像绝对路径,
                     'cam_intrinsic': (4,4) float64,
                     'sensor2lidar_rotation': (3,3),
                     'sensor2lidar_translation': list[3],
                     'cam_distorted': (14,) 或 (4,),
                     'ego2img': (4,4),
                     ...
                   }

相机映射 (前视 / 左前 / 右前):
  cam1 -> cam_f0
  cam4 -> cam_l0
  cam8 -> cam_r0

需要你手动填的接口:
  - DRIVING_COMMAND  (4 维 one-hot,[左, 直, 右, 未知])
  - EGO_VELOCITY     (vx, vy, m/s)
  - EGO_ACCELERATION (ax, ay, m/s^2)

用法:
  python phigent_test/run_inference.py \
      --manifest <path> --index 0 \
      --checkpoint ckpt/sparsedrive_navsimv1.ckpt
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# 用户接口:导航命令 / 自车运动状态(留给你自己填)
# ---------------------------------------------------------------------------
DRIVING_COMMAND = np.array([0, 1, 0, 0], dtype=np.int64)   # [left, straight, right, unknown]
EGO_VELOCITY = np.array([8., 0.0], dtype=np.float32)      # (vx, vy) in ego frame, m/s
EGO_ACCELERATION = np.array([0.0, 0.0], dtype=np.float32)  # (ax, ay) in ego frame, m/s^2

# phigent cam id -> SparseDrive 命名
CAM_MAP = {
    "cam_f0": "cam1",  # front
    "cam_l0": "cam4",  # front-left
    "cam_r0": "cam8",  # front-right
}

DEFAULT_MANIFEST = (
    "/mnt/cfs-baidu/public/jiahao.chen/all_guideline_dataset_iid_unimap_0512/2005309951507853312/17A1783S0_20241226T155307_17A1783S0_20241226T155307_downsample_0_1735199647388_1735199707388.pkl"
)
DEFAULT_CHECKPOINT = "ckpt/sparsedrive_navsimv1.ckpt"


# ---------------------------------------------------------------------------
# 准备 sys.path,确保可以 import navsim
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

warnings.filterwarnings("ignore")  # shapely <2.0 unpickle warning


def load_phigent_frame(manifest_path: str, index: int) -> dict:
    with open(manifest_path, "rb") as f:
        manifest = pickle.load(f)
    entry = manifest[index]
    print(f"[load] manifest entry[{index}].iid = {entry['iid']}")
    with open(entry["iid"], "rb") as f:
        sample = pickle.load(f)
    return sample[0] if isinstance(sample, list) else sample


def build_features(frame: dict, config) -> dict:
    """
    把 phigent frame -> SparseDriveModel.forward 直接吃的 features dict。

    绕开 navsim 的 Cameras dataclass / SceneLoader,自己照着
    SparseDriveFeatureBuilder 的输出格式拼。
    """
    from navsim.agents.sparsedrive.sparsedrive_features import SparseDriveFeatureBuilder

    # --- 把 phigent cams 翻译成 SparseDrive 期望的 camera_feature 列表 ---
    # 格式参考 sparsedrive_features.py:_get_camera_feature 的输出。
    # 模型只用最后一帧 (history 的 -1),这里只塞当前帧。
    camera_keys = list(CAM_MAP.keys())  # cam_f0, cam_l0, cam_r0
    frame_info = {key: {} for key in camera_keys}

    for sd_name, phi_name in CAM_MAP.items():
        cam = frame["cams"][phi_name]
        s2l_rot = np.asarray(cam["sensor2lidar_rotation"], dtype=np.float64)         # (3,3)
        s2l_t = np.asarray(cam["sensor2lidar_translation"], dtype=np.float64)        # (3,)
        K = np.asarray(cam["cam_intrinsic"], dtype=np.float64)                       # (4,4) or (3,3)
        # SparseDrive feature builder 期望 intrinsics 形状用法见 get_camera_params:
        # viewpad = eye(4); viewpad[:K.shape[0], :K.shape[1]] = K。所以 (4,4) 直接用,(3,3) 也兼容。
        distortion = np.asarray(cam["cam_distorted"], dtype=np.float32)

        frame_info[sd_name]["image_path"] = cam["data_path"]
        frame_info[sd_name]["sensor2lidar_rotation"] = s2l_rot
        frame_info[sd_name]["sensor2lidar_translation"] = s2l_t
        frame_info[sd_name]["intrinsics"] = K
        frame_info[sd_name]["distortion"] = distortion

    fb = SparseDriveFeatureBuilder(config)

    # 复用 fb 的图像 + 几何处理流水线(test_mode=True 关掉所有随机增广)
    results = fb.get_camera_params(frame_info)
    results = fb.load_images(results)
    results = fb.resize_crop_flip_img(results, test_mode=True)
    # ego_rotation 在 test_mode 下 angle=0,直接跳过
    # photo_metric_distortion 在 test_mode 下直接 return
    results = fb.normalize_img(results)
    results = fb.data_adapter(results)

    # --- status_feature: [driving_command(4), ego_velocity(2), ego_acceleration(2)] ---
    status_feature = torch.cat([
        torch.tensor(DRIVING_COMMAND, dtype=torch.float32),
        torch.tensor(EGO_VELOCITY, dtype=torch.float32),
        torch.tensor(EGO_ACCELERATION, dtype=torch.float32),
    ])

    # --- 打 batch 维 ---
    camera_feature = {}
    for k, v in results.items():
        if isinstance(v, torch.Tensor):
            camera_feature[k] = v.unsqueeze(0)
        elif isinstance(v, np.ndarray):
            camera_feature[k] = torch.from_numpy(v).unsqueeze(0)
        else:
            camera_feature[k] = v

    features = {
        "camera_feature": camera_feature,
        "status_feature": status_feature.unsqueeze(0),
    }
    return features


def project_traj_to_image(traj_xy: np.ndarray, lidar2img: np.ndarray,
                          img_w: int, img_h: int, z: float = 0.0) -> np.ndarray:
    """
    把 ego 系下的 (N, 2) 轨迹点用 lidar2img 投到图像像素。
    返回 (N, 2) 像素坐标 (u, v),投到图外 / 后方的点用 NaN。
    源码 (sparsedrive_features.get_camera_params) 里的约定:
        lidar2img = K_pad @ lidar2cam.T
    其中 .T 已经施加在 lidar2cam 上,所以投影是标准的左乘:
        pix_h = lidar2img @ pts_h
    """
    N = traj_xy.shape[0]
    pts = np.concatenate([traj_xy, np.full((N, 1), z), np.ones((N, 1))], axis=1)  # (N, 4)
    pix_h = (lidar2img @ pts.T).T  # (N, 4)
    z_cam = pix_h[:, 2]
    valid = z_cam > 1e-3
    uv = np.full((N, 2), np.nan, dtype=np.float32)
    uv[valid, 0] = pix_h[valid, 0] / pix_h[valid, 2]
    uv[valid, 1] = pix_h[valid, 1] / pix_h[valid, 2]
    in_img = (uv[:, 0] >= 0) & (uv[:, 0] < img_w) & (uv[:, 1] >= 0) & (uv[:, 1] < img_h)
    uv[~in_img] = np.nan
    return uv


def visualize(features: dict, trajectory: np.ndarray,
              save_path: Path | None = None,
              frame: dict | None = None,
              cam_order=("cam_l0", "cam_f0", "cam_r0"),
              img_mean=(123.675, 116.28, 103.53),
              img_std=(58.395, 57.12, 57.375),
              return_array: bool = False,
              dpi: int = 100):
    """
    上排 3 路相机,下排 BEV。前视图直接用 phigent 原始图 + 原始
    ego2img 矩阵(K @ inv(sensor2ego))做投影,避免 pipeline 缩放/裁剪
    带来的累积误差。左前 / 右前两路只展示 pipeline 处理过的归一化图。

    :param frame: 原始 phigent frame dict(必须传入,用于读取 cam1 原图和 ego2img)
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    cf = features["camera_feature"]
    imgs = cf["imgs"][0].detach().cpu().numpy()              # (3, 3, 256, 512)

    mean = np.array(img_mean, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(img_std, dtype=np.float32).reshape(3, 1, 1)

    rgbs = []
    for i in range(imgs.shape[0]):
        img = imgs[i] * std + mean
        img = np.clip(img, 0, 255).astype(np.uint8)
        rgbs.append(np.transpose(img, (1, 2, 0)))

    traj_xy = trajectory[:, :2]  # (8, 2)

    # 前视原图 + 原始 ego2img(直接来自 frame['cams']['cam1'])
    f0_img = None
    f0_ego2img = None
    if frame is not None:
        cam_f = frame["cams"][CAM_MAP["cam_f0"]]
        f0_img = np.array(Image.open(cam_f["data_path"]))
        f0_ego2img = np.asarray(cam_f["ego2img"], dtype=np.float64)

    # ----------- 画图 -----------
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.4], hspace=0.15, wspace=0.05)
    titles = {"cam_l0": "front-left (cam4)",
              "cam_f0": "front (cam1)",
              "cam_r0": "front-right (cam8)"}
    for i, name in enumerate(cam_order):
        ax = fig.add_subplot(gs[0, i])

        if name == "cam_f0" and f0_img is not None:
            # 用原始图 + 原始 ego2img 投影
            H, W = f0_img.shape[:2]
            ax.imshow(f0_img)
            uv = project_traj_to_image(traj_xy, f0_ego2img, W, H)
            valid = ~np.isnan(uv[:, 0])
            if valid.sum() >= 2:
                ax.plot(uv[valid, 0], uv[valid, 1], "-", color="lime", linewidth=3)
            ax.scatter(uv[valid, 0], uv[valid, 1], s=40, c="red", zorder=3)
        else:
            # 其它路只展示 pipeline 处理过的 256x512 归一化图
            ax.imshow(rgbs[i])

        ax.set_title(titles.get(name, name))
        ax.set_xticks([])
        ax.set_yticks([])

    # BEV(下排,跨 3 列)
    ax = fig.add_subplot(gs[1, :])
    # ego 系: +x 前, +y 左 -> matplotlib 上,横轴 = -y(让左侧画在左边),纵轴 = x(向上)
    xs = traj_xy[:, 0]
    ys = traj_xy[:, 1]
    ax.plot(-ys, xs, "-o", color="tab:blue", linewidth=2, markersize=4, label="pred traj")
    # 自车朝前的箭头
    ax.add_patch(Rectangle((-0.9, -2.4), 1.8, 4.8, fill=False, edgecolor="k", linewidth=1.5))
    ax.arrow(0, 0, 0, 1.5, head_width=0.4, head_length=0.4, fc="k", ec="k")
    ax.set_xlabel("-y (m)   <- left | right ->")
    ax.set_ylabel("x (m)   forward ->")
    ax.set_title("BEV (ego frame)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    # 横纵坐标范围统一,以轨迹最大尺度为准
    rng = max(10.0, float(np.max(np.abs(traj_xy))) * 1.2 + 2)
    ax.set_xlim(-rng, rng)
    ax.set_ylim(-rng, rng)
    ax.legend(loc="upper right")

    fig.suptitle(f"SparseDriveV2 prediction  |  cmd={DRIVING_COMMAND.tolist()}  "
                 f"v={EGO_VELOCITY.tolist()} m/s  a={EGO_ACCELERATION.tolist()} m/s²",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    arr = None
    if return_array:
        # 渲染到内存 buffer,转成 (H, W, 3) uint8
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        arr = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return arr


def _infer_version(checkpoint_path: str) -> str:
    """从 ckpt 文件名推断 v1/v2(navsimv1/navsimv2)。"""
    name = Path(checkpoint_path).name.lower()
    if "navsimv2" in name or "_v2" in name:
        return "v2"
    if "navsimv1" in name or "_v1" in name:
        return "v1"
    return "v1"  # 默认 v1


def load_agent(checkpoint_path: str, version: str = "auto"):
    """
    version: 'v1' / 'v2' / 'auto'(默认从 checkpoint_path 文件名推断)。

    v1 与 v2 的差异(参考 scripts/training/sparsedrive_navsimv1.sh / v2.sh
    以及 custom_decoder.py 里 v1/v2 分支的指标聚合):
      - dataset_version
      - metric_heads 集合(影响 ckpt 的 state_dict key)
      - velocity_filter_num: v1=(64,20), v2=(64,10)(SparseDriveConfig 默认值)
    """
    from navsim.agents.sparsedrive.sparsedrive_agent import SparseDriveAgent
    from navsim.agents.sparsedrive.sparsedrive_config import SparseDriveConfig

    if version == "auto":
        version = _infer_version(checkpoint_path)
    assert version in ("v1", "v2"), f"unknown version {version}"

    config = SparseDriveConfig()
    config.cams = ("cam_l0", "cam_f0", "cam_r0")

    if version == "v1":
        config.dataset_version = "v1"
        config.metrics = (
            "no_at_fault_collisions",
            "drivable_area_compliance",
            "driving_direction_compliance",
            "time_to_collision_within_bound",
            "comfort",
            "ego_progress",
        )
        config.velocity_filter_num = (64, 20)
    else:  # v2
        # SparseDriveConfig 的默认值就是 v2 训练脚本的设置,这里显式列出便于调试
        config.dataset_version = "v2"
        config.metrics = (
            "no_at_fault_collisions",
            "drivable_area_compliance",
            "driving_direction_compliance",
            "traffic_light_compliance",
            "time_to_collision_within_bound",
            "ego_progress",
            "lane_keeping",
            "history_comfort",
        )
        config.velocity_filter_num = (64, 10)

    print(f"[model] version={version}, metrics={config.metrics}")

    agent = SparseDriveAgent(
        config=config,
        lr=0.0,
        checkpoint_path=checkpoint_path,
    )
    agent.initialize()
    agent.eval()
    return agent, config


def run_one_frame(frame: dict, agent, config, device: str):
    """单帧:构造 features -> forward -> 返回 (features, trajectory)。"""
    features = build_features(frame, config)
    features["status_feature"] = features["status_feature"].to(device)
    cf = features["camera_feature"]
    for k, v in cf.items():
        if isinstance(v, torch.Tensor):
            cf[k] = v.to(device)
    with torch.no_grad():
        pred, _ = agent._sparsedrive_model(features, targets=None)
    trajectory = pred["trajectory"][0].cpu().numpy()
    return features, trajectory


def run_video(manifest_path: str, agent, config, device: str,
              video_path: Path, fps: int = 10, max_frames: int | None = None):
    """跑整个 manifest -> 10Hz mp4。"""
    import cv2

    with open(manifest_path, "rb") as f:
        manifest = pickle.load(f)
    n = len(manifest) if max_frames is None else min(len(manifest), max_frames)
    print(f"[video] {n} frames @ {fps}fps -> {video_path}")

    writer = None
    video_path.parent.mkdir(parents=True, exist_ok=True)

    for i in range(n):
        with open(manifest[i]["iid"], "rb") as f:
            sample = pickle.load(f)
        frame = sample[0] if isinstance(sample, list) else sample

        features, trajectory = run_one_frame(frame, agent, config, device)
        img_rgb = visualize(features, trajectory, save_path=None,
                            frame=frame, return_array=True)

        # OpenCV 要 BGR
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        if writer is None:
            h, w = img_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"cv2.VideoWriter 打不开 {video_path}, 试试 .avi 后缀")
            print(f"[video] frame size = {w}x{h}")
        else:
            # 帧尺寸偶尔会因为 figure 内容变化(BEV 范围)略不同 -> resize 对齐
            if (img_bgr.shape[1], img_bgr.shape[0]) != (w, h):
                img_bgr = cv2.resize(img_bgr, (w, h))
        writer.write(img_bgr)

        if (i + 1) % 10 == 0 or i == n - 1:
            print(f"[video] {i + 1}/{n}")

    if writer is not None:
        writer.release()
    print(f"[video] done -> {video_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--index", type=int, default=0,
                        help="单帧模式下 manifest 中要跑的样本序号")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--version", default="auto", choices=["auto", "v1", "v2"],
                        help="模型版本 (v1=PDMS / v2=EPDMS)。auto 时从 ckpt 文件名推断。")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save", default="phigent_test/pred_trajectory.npy",
                        help="单帧模式:预测轨迹 (8, 3) 输出 npy 路径")
    parser.add_argument("--save-vis", default="phigent_test/pred_visualization.png",
                        help="单帧模式:可视化 PNG 输出路径")
    # --- 视频模式 ---
    parser.add_argument("--video", default=None,
                        help="给定路径(.mp4/.avi)则跑整个 manifest 并合成视频")
    parser.add_argument("--fps", type=int, default=10,
                        help="视频帧率(默认 10Hz)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="视频模式:最多跑前 N 帧(默认全跑)")
    args = parser.parse_args()

    # 加载模型(单帧 / 视频共用)
    print(f"[model] loading {args.checkpoint} on {args.device}")
    agent, config = load_agent(args.checkpoint, version=args.version)
    agent = agent.to(args.device)

    if args.video is not None:
        run_video(args.manifest, agent, config, args.device,
                  Path(args.video), fps=args.fps, max_frames=args.max_frames)
        return

    # ---------- 单帧模式 ----------
    frame = load_phigent_frame(args.manifest, args.index)
    print(f"[load] frame token = {frame.get('token')}, "
          f"available cams = {list(frame['cams'].keys())}")

    print("[infer] forward ...")
    features, trajectory = run_one_frame(frame, agent, config, args.device)

    cf = features["camera_feature"]
    print(f"       imgs.shape          = {tuple(cf['imgs'].shape)}")
    print(f"       projection_mat.shape= {tuple(cf['projection_mat'].shape)}")
    print(f"       status_feature      = {features['status_feature'].cpu().numpy()[0]}")

    print("\n[result] predicted ego trajectory (next 4s, 0.5s interval), ego frame:")
    print("    step   x[m]    y[m]    yaw[rad]")
    for i, (x, y, yaw) in enumerate(trajectory):
        print(f"    {i:>3d}  {x:>7.3f} {y:>7.3f}  {yaw:>+7.3f}")

    out = Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, trajectory)
    print(f"\n[save] trajectory -> {out}")

    vis_path = Path(args.save_vis)
    print(f"[vis] rendering -> {vis_path}")
    visualize(features, trajectory, vis_path, frame=frame)
    print(f"[save] visualization -> {vis_path}")


if __name__ == "__main__":
    main()

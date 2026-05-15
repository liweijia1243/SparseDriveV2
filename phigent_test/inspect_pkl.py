"""
读取并探查 phigent 数据集的 pkl 索引文件。

用法:
    python inspect_pkl.py
    python inspect_pkl.py --manifest <path-to-manifest.pkl> --index 0

在 VSCode 中打断点调试:
    1. 打开本文件,在你想停下来的行号左侧灰色区域单击 -> 出现红点
       推荐断点位置:
         - load_manifest 返回后 (查看 manifest 列表)
         - load_frame 返回后 (查看 frame dict 的 50+ 字段)
         - main 中 frame = load_frame(...) 之后 (查看具体字段)
    2. 按 F5 启动调试 (首次会让你选择 "Python File" 配置)
    3. 命中断点后,在左侧 "VARIABLES" 面板展开 frame / manifest 查看
       或在 "DEBUG CONSOLE" 输入表达式:
           frame.keys()
           frame['cams'].keys()
           frame['gt_ego_traj']
           type(frame['gt_object_instances'])
"""
import argparse
import os
import pickle
from pprint import pprint

import numpy as np


DEFAULT_MANIFEST = (
    "/mnt/cfs-baidu/public/jiahao.chen/all_guideline_dataset_iid_unimap_0512/"
    "2005931423737794560/"
    "17A1783S0_20241223T152657_17A1783S0_20241223T152657"
    "_downsample_0_1734939056948_1734939116927.pkl"
)


def load_manifest(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_frame(iid_path):
    with open(iid_path, "rb") as f:
        sample = pickle.load(f)
    return sample[0] if isinstance(sample, list) else sample


def describe(value, name="root", depth=0, max_depth=2):
    pad = "  " * depth
    if isinstance(value, dict):
        print(f"{pad}{name}: dict ({len(value)} keys)")
        if depth < max_depth:
            for k, v in value.items():
                describe(v, repr(k), depth + 1, max_depth)
    elif isinstance(value, np.ndarray):
        print(f"{pad}{name}: ndarray shape={value.shape} dtype={value.dtype}")
    elif isinstance(value, (list, tuple)):
        print(f"{pad}{name}: {type(value).__name__} len={len(value)}")
        if value and depth < max_depth:
            describe(value[0], f"{name}[0]", depth + 1, max_depth)
    else:
        s = repr(value)
        if len(s) > 120:
            s = s[:120] + "..."
        print(f"{pad}{name}: {type(value).__name__} = {s}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST,
                        help="顶层 manifest pkl 路径")
    parser.add_argument("--index", type=int, default=0,
                        help="从 manifest 中取第几条样本来加载")
    parser.add_argument("--max-depth", type=int, default=2,
                        help="describe() 递归深度")
    args = parser.parse_args()

    print(f"[1] loading manifest: {args.manifest}")
    manifest = load_manifest(args.manifest)
    print(f"    manifest: list len={len(manifest)}")
    print(f"    entry[0] keys: {list(manifest[0].keys())}")

    entry = manifest[args.index]
    iid = entry["iid"]
    extra_dir = entry["extra_annotation_path"]
    print(f"\n[2] entry[{args.index}]:")
    print(f"    iid = {iid}")
    print(f"    extra_annotation_path = {extra_dir} (is_dir={os.path.isdir(extra_dir)})")

    print(f"\n[3] loading frame from iid ...")
    frame = load_frame(iid)
    print(f"    frame: dict ({len(frame)} keys)")
    print(f"    keys: {list(frame.keys())}")

    print("\n[4] structural overview (max_depth={}):".format(args.max_depth))
    describe(frame, name="frame", max_depth=args.max_depth)

    # === 在这里打断点最方便,可以在 VSCode 调试器里随意探索 frame ===
    # 比如试试这些表达式:
    #   frame['cams']
    #   frame['gt_ego_traj']
    #   [k for k in frame if k.startswith('gt_')]
    print("\n[done] set a breakpoint on this line to inspect `frame` interactively.")


if __name__ == "__main__":
    main()

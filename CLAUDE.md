# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

This is the SparseDriveV2 fork of [NAVSIM](https://github.com/autonomousvision/navsim) — a scoring-based end-to-end autonomous driving planner evaluated on the NAVSIM benchmark (v1 and v2) and Bench2Drive (separate `bench2drive` branch). The package installs as `navsim` (see `setup.py`); the SparseDriveV2 agent lives under `navsim/agents/sparsedrive/`.

## Environment

The codebase assumes the upstream NAVSIM environment is already installed (see the upstream README). Three env vars are required by all run scripts and must be set in your shell:

- `NAVSIM_DEVKIT_ROOT` — repo root (this directory)
- `NAVSIM_EXP_ROOT` — output dir for caches/checkpoints/logs (scripts hard-code `exp/...` relative paths in some cases and `$NAVSIM_EXP_ROOT/...` in others — both conventions are mixed; check the specific script)
- `OPENSCENE_DATA_ROOT` — data root containing `navtrain/`, `navtest/`, `navhard_two_stage/`

After installing NAVSIM, install the SparseDrive custom CUDA op:

```bash
conda activate navsim
cd navsim/agents/sparsedrive/ops
python setup.py develop
```

This builds `deformable_aggregation_ext` (and `_with_depth_ext`) — the deformable feature aggregation kernels that the sparse decoder relies on. Prebuilt `.so` files are checked in but typically need to be rebuilt against the local CUDA/PyTorch.

Anchor files (`ckpt/kmeans/path_*.npy`, `velocity_*.npy`, `trajectory_*.npz`) and the ResNet-34 backbone (`ckpt/resnet34.bin`) must be present before training/eval — download from the project's HuggingFace or run `scripts/cluster/cluster_anchor.py`.

## Common workflows

All commands assume `conda activate navsim` and the env vars above are set.

### Data and metric caching (run once before training/eval)

```bash
sh scripts/cache/run_dataset_caching_navtrain.sh
sh scripts/cache/run_dataset_caching_navtest.sh

# v1 metrics (PDMS)
sh scripts/cache/run_metric_caching_navtrain_v1.sh
sh scripts/cache/run_metric_caching_navtest_v1.sh

# v2 metrics (EPDMS)
sh scripts/cache/run_metric_caching_navtrain_v2.sh
sh scripts/cache/run_metric_caching_navtest_v2.sh
```

### Training

```bash
sh scripts/training/sparsedrive_navsimv1.sh   # navsimv1 config
sh scripts/training/sparsedrive_navsimv2.sh   # navsimv2 config
```

Training is PyTorch Lightning + Hydra. `run_training.py` reads `navsim/planning/script/config/training/default_training.yaml`; the agent yaml lives at `navsim/planning/script/config/common/agent/sparsedrive_agent.yaml`. Override anything from CLI: e.g. `dataloader.params.batch_size=8 trainer.params.max_epochs=20 agent.lr=5e-5`. Note the v1 vs v2 scripts differ by `+agent.config.dataset_version`, `+agent.config.metrics=[...]`, and `+agent.config.velocity_filter_num=[...]` overrides — when adding a new training variant, follow that pattern rather than editing the yaml.

### Evaluation

```bash
sh scripts/evaluation/run_pdm_score_navtest_v1.sh    # PDMS (v1)
sh scripts/evaluation/run_pdm_score_navtest_v2.sh    # EPDMS (v2) — reports both pre- and post-bug-fix scores
sh scripts/evaluation/run_pdm_score_navhard.sh       # navhard two-stage
```

Each eval script points at a checkpoint (`CHECKPOINT=ckpt/...ckpt`) and a metric cache (`CACHE_PATH=...`); update both when running a new model. The v2 evaluator deliberately reports both EPDMS values around the upstream bug fix (see comment in `docs/train_eval.md`).

## Architecture

### How an agent plugs in

The pipeline is the upstream NAVSIM contract: `AbstractAgent` (`navsim/agents/abstract_agent.py`) → `AgentLightningModule` (`navsim/planning/training/agent_lightning_module.py`) → Hydra `run_training.py` / `run_pdm_score_*.py`. An agent supplies:

1. `get_sensor_config()` — which cameras/lidar to load
2. `get_feature_builders()` / `get_target_builders()` — turn raw `AgentInput`/`Scene` into tensors (and during caching, write them to disk)
3. `get_training_callbacks()` and `forward()` — standard Lightning training
4. `compute_trajectory()` — inference path used by the PDM scorer

`SparseDriveAgent` (`navsim/agents/sparsedrive/sparsedrive_agent.py`) is the entry point — it wires `SparseDriveModel`, `SparseDriveFeatureBuilder`, `SparseDriveTargetBuilder`, and `CheckpointCallback`.

### SparseDrive model layout

`navsim/agents/sparsedrive/`:
- `sparsedrive_config.py` — single source of truth for hyperparameters (anchor paths, vocabulary sizes `mode_path=1024`/`mode_vel=256`, decoder layers, deformable head, metric supervision settings). Hydra overrides flow through here.
- `sparsedrive_backbone.py` — ResNet-34 + FPN producing multi-scale image features.
- `sparsedrive_model.py` — `SparseDriveModel` (backbone + status encoder + `TrajectoryHead`).
- `custom_decoder.py`, `blocks.py` — sparse transformer decoder over path/velocity/trajectory queries.
- `ops/deformable_aggregation.py` (+ `src/`) — custom CUDA op for deformable feature aggregation; build via `ops/setup.py`.
- `scorer/` — PDM scoring used as supervision and at eval (`get_pdm_score_v1.py`/`_v2.py`, `pdm_score_v1.py`/`_v2.py`). The scorer drives metric-based loss in training.

The factorized vocabulary (paths × velocity profiles → composed trajectories) is the core idea — anchor files in `ckpt/kmeans/` are not optional, and `mode_path`, `mode_vel`, `path_filter_num`, `velocity_filter_num` together control the coarse-then-fine scoring funnel.

### Caching is mandatory

Both data features and PDM metrics are cached to disk before training. Training scripts pass `use_cache_without_dataset=True force_cache_computation=False cache_path=exp/data_cache_navtrain` — meaning the dataset class loads directly from cached tensors and never touches the raw scene pickles. If you change `SparseDriveFeatureBuilder` / `SparseDriveTargetBuilder` outputs, you must re-run dataset caching, otherwise old cached tensors will silently mismatch the new builder shapes.

### NAVSIM v1 vs v2

v1 uses PDMS with 6 metrics; v2 uses EPDMS with 8 metrics (adds `traffic_light_compliance`, `lane_keeping`, `history_comfort`, swaps `comfort` semantics). The agent config's `dataset_version`, `metrics`, and `velocity_filter_num` all change between the two — keep the v1 and v2 scripts as separate entry points rather than trying to unify them.

## Tutorial / visualization

`tutorial/tutorial_visualization.ipynb` is the reference for visualizing predictions and renders the per-frame GIFs at the repo root. Use it (rather than ad-hoc plotting code) when adding new visualization helpers.

#!/usr/bin/env bash
# Debug-only v2 training entrypoint over the single-log debug_mini split.
# Prereqs:
#   1. sh scripts/cache/run_dataset_caching_debug_mini.sh
#   2. sh scripts/cache/run_metric_caching_debug_mini_v2.sh
#   3. export OPENSCENE_DATA_ROOT=/mnt/cfs-baidu/public/jiahao.chen/navsim_workspace/dataset
# num_workers=0 so breakpoints fire in the main process.

export HYDRA_FULL_ERROR=1

config=default_training
agent=sparsedrive_agent

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
    --config-name $config \
    agent=$agent \
    experiment_name=${agent}_debug_mini_v2 \
    train_test_split=debug_mini \
    use_cache_without_dataset=True \
    force_cache_computation=False \
    cache_path=$NAVSIM_EXP_ROOT/data_cache_navtrain_debug_mini \
    dataloader.params.batch_size=2 \
    dataloader.params.num_workers=0 \
    dataloader.params.prefetch_factor=null \
    trainer.params.max_epochs=1 \
    agent.lr=0.0001

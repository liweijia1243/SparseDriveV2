#!/usr/bin/env bash
# Build the SparseDrive feature/target cache for the debug_mini split.
# Output dir basename MUST contain "data_cache_navtrain" so that
# custom_decoder.py:274's str.replace(...) can map token paths to the
# matching metric_cache directory at training time.

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching.py \
    agent=sparsedrive_agent \
    experiment_name=cache_debug_mini \
    train_test_split=debug_mini \
    cache_path=$NAVSIM_EXP_ROOT/data_cache_navtrain_debug_mini

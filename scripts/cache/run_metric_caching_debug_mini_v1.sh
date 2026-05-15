#!/usr/bin/env bash
# Build the v1 PDM metric cache for the debug_mini split.
# Output dir basename MUST be "metric_cache_navtrainv1<suffix>" matching the
# dataset cache "data_cache_navtrain<suffix>" — see custom_decoder.py:274.

TRAIN_TEST_SPLIT=debug_mini
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache_navtrainv1_debug_mini

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching_v1.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    cache.cache_path=$CACHE_PATH

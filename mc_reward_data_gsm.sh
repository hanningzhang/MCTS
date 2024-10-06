#!/bin/bash
export VLLM_CACHE_ROOT='/tmp/rpan2_vllm_cache_s-$1.r-$2'
export XDG_CACHE_HOME='/tmp/rpan2_xdg_cache_s-$1.r-$2'
export OUTLINES_CACHE_DIR='/tmp/rpan2_outline_cache_s-$1.r-$2'
python mc_reward_data_gsm.py \
    --completion_model_name_or_path deepseek-ai/deepseek-math-7b-rl \
    --dataset_path HanningZhang/gsm-deepseek \
    --output_dir smooth_reward_data \
    --tensor_parallel_size 1 \
    --num_gpus 1 \
    --local_rank $1 \
    --sampling_num 16 \
    --split $2 

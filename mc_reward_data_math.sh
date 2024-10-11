#!/bin/bash
export VLLM_CACHE_ROOT='/tmp/${USER}_vllm_cache_s-$1.r-$2_math'
export XDG_CACHE_HOME='/tmp/${USER}_xdg_cache_s-$1.r-$2_math'
export OUTLINES_CACHE_DIR='/tmp/${USER}_outline_cache_s-$1.r-$2_math'

mkdir -p tmp
mark=tmp/math-$1-$2

if [ -f ${mark}.ongoing -o -f ${mark}.complete ]; then
  echo "$(date):    skip ${mark}..."
  exit 0
fi
trap "rm -f ${mark}.ongoing; exit" SIGINT SIGTERM SIGKILL
touch ${mark}.ongoing

python mc_reward_data_math.py \
    --completion_model_name_or_path deepseek-ai/deepseek-math-7b-rl \
    --dataset_path HanningZhang/math-deepseek \
    --output_dir /scratch/bdjz/rpan2/smooth_reward_data \
    --tensor_parallel_size 1 \
    --num_gpus 4 \
    --local_rank $1 \
    --sampling_num 16 \
    --split $2 \
    --batch_size 50 \
    --num_batches_per_save 1

if [ $? -eq 0 ]; then
  touch ${mark}.complete
fi

rm ${mark}.ongoing

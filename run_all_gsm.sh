#!/bin/bash

exp=gsm;
for id in 0 1 2 3; do for split in 0 1; do
  srun \
    --account=bckr-delta-gpu \
    --partition=gpuA100x4 \
    --nodes=1 \
    --gpus-per-node=1 \
    --tasks=1 \
    --tasks-per-node=1 \
    --cpus-per-task=32 \
    --mem=128g \
    --time=48:00:00 \
    bash mc_reward_data_${exp}.sh ${id} ${split} \
    > log/${exp}-${id}-${split}.log \
    2> log/${exp}-${id}-${split}.err &
  done
done
#!/bin/bash

exp_name="count-data"
sbatch \
    --job-name="${exp_name}" \
    --account=bckr-delta-gpu \
    --partition=gpuMI100x8 \
    --nodes=1 \
    --gpus-per-node=0 \
    --tasks=1 \
    --tasks-per-node=1 \
    --cpus-per-task=4 \
    --mem=128g \
    --time=48:00:00 \
    --output="log/${exp_name}.log" \
    --error="log/${exp_name}.err" \
    ./count_data.sh smooth_reward_data \
    > log/sh_${exp_name}.log \
    2> log/sh_${exp_name}.err &

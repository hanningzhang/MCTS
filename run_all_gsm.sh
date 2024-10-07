#!/bin/bash

exp=gsm;
for id in 0 1 2 3; do for split in 0 1; do
  exp_name="${exp}-${id}-${split}"
  if [ -f tmp/${exp_name}.ongoing ]; then
    continue
  fi
  echo "$(date): ${exp_name}"
  sbatch \
    --job-name="${exp}-${id}-${split}" \
    --account=bckr-delta-gpu \
    --partition=gpuA40x4 \
    --nodes=1 \
    --gpus-per-node=1 \
    --tasks=1 \
    --tasks-per-node=1 \
    --cpus-per-task=32 \
    --mem=128g \
    --time=48:00:00 \
    --output="log/${exp_name}.log" \
    --error="log/${exp_name}.err" \
    ./mc_reward_data_${exp}.sh ${id} ${split} \
    > log/sh_${exp_name}.log \
    2> log/sh_${exp_name}.err &
    sleep 1
  done
done

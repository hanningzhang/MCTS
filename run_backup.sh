#!/bin/bash

src_dir=/scratch/bdjz/rpan2/smooth_reward_data/
dst_dir=/scratch/bckr/rpan2/smooth_reward_data.backup/

# Backups
while true; do
  echo "FILENAME  SRC_SIZE   BAKCUP_SIZE"
  for file in ${src_dir}/*; do
    filename=$(basename ${file})
    src_size=$(du ${file} | awk '{ print $1 }')
    dst_file=${dst_dir}/${filename}
    if [ -f "${dst_file}" ]; then
      dst_size=$(du ${dst_file} | awk '{ print $1 }')
    else
      dst_size=0
    fi
    echo "${filename} ${src_size} ${dst_size}"
    if [ ${src_size} -gt ${dst_size} ]; then
      cp ${file} ${dst_file}
    fi
  done
  sleep 600
done

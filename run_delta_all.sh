#!/bin/bash
#

./run_delta_slurm.sh -s run_gsm_split0.sh

./run_delta_slurm.sh -s run_gsm_split1.sh

./run_delta_slurm.sh -s run_math_split0.sh

./run_delta_slurm.sh -s run_math_split1.sh






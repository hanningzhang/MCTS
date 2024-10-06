#!/bin/bash

partition=gpuA40x4
gpus_per_node=4
cpus_per_task=32
mem='512G'
time_s='48:00:00'
ba_script='run.sh'


# set email
email=${USER@}illinois.edu

usage() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -p gpuA40x4|gpuA100x4    Partition to use"
    echo "  -g 1|2|4    Number of GPUs per node"
    echo "  -c 16|24    Number of CPU cores per task"
    echo "  -m 64g|128g Memory allocation"
    echo "  -s bash_script.sh The bash script to run
    echo "  -t 08:00:00|24:00:00|48:00:00   Time duration hh:mm:ss (48 hour limit)"
    echo "  -h    Display this help message and exit"
    echo
    exit 1
}

 
# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
  case "$1" in
     -p|--partition) partition="$2"; shift 2;;
     -g|--gpus) gpus_per_node="$2"; shift 2;;
     -c|--cpus) cpus_per_task="$2"; shift 2;;
     -m|--mem) mem="$2"; shift 2;;
     -s|--script) ba_script="$2"; shift 2;;
     -t|--time) time_s="$2"; shift 2;;
     -h|--help) usage ;;
     *) echo "Unknown option: $1"; usage ;;
  esac
done

# set jobname
jobname="${ba_script_$(date +"%Y-%m-%d-%H-%M-%S")"

# set account
hostname=$(hostname)
if [[ "$hostname" == *"gh"* ]]; then
    account="bckr-dtai-gh"
else
    account="bckr-delta-gpu"
fi

# set work_dir
work_dir=`pwd`
echo working dir is ${work_dir}

# Create a temporary SLURM scrpt with arguments

temp_sbatch_script=/tmp/temp_sbatch_script.sh

cat << EOT > ${temp_sbatch_script}
#!/bin/bash
#SBATCH --account=${account}
#SBATCH --job-name=${jobname}
#SBATCH --output=logs/${jobname}.log
#SBATCH --partition=${partition}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=${gpus_per_node}
#SBATCH --gpu-bind=none 
#SBATCH --cpus-per-task=${cpus_per_task}
#SBATCH --mem=${mem}
#SBATCH --time=${time_s}
#SBATCH --mail-type=END
#SBATCH --mail-user=${email}

# Load necessary modules
#module load python

cd ${work_dir}

bash ${ba_script}

EOT


# display submit and remove
echo === begin sbatch script ===
cat ${temp_sbatch_script}
echo === end sbatch script ===
sbatch ${temp_sbatch_script}
/bin/rm ${temp_sbatch_script}

echo on

sleep 2

squeue -u $USER


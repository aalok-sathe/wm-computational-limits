#!/bin/bash
#SBATCH --mail-user=chmosky@duck.com
#SBATCH --mail-type=ALL

#SBATCH --mem-per-cpu 64G

# Request a GPU partition node and access to 1 GPU
#SBATCH -p {slurm_partition_argument} --gres=gpu:1

#SBATCH -a 1-10%15
#SBATCH -t 1-00:00:00

#SBATCH -o {batch_output_prefix}batch_output/training_run_%A_%a.out
#SBATCH --nodes=1 

set -x

. .venv/bin/activate
echo "find sample run at batch-output/training_run_${{SLURM_ARRAY_JOB_ID}}_1.out"

sleep $((RANDOM % 30 + 1))

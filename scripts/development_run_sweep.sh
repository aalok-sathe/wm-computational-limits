#!/bin/bash
#SBATCH --mail-user=chmosky@duck.com
#SBATCH --mail-type=ALL

#SBATCH --mem-per-cpu 64G

# Request a GPU partition node and access to 1 GPU
#SBATCH -p 3090-gcondo --gres=gpu:1

#SBATCH -a 1-1%15
#SBATCH -t 2-00:00:00

#SBATCH -o batch-output/training_run_%A_%a.out
#SBATCH --nodes=1 

set -x

. .venv/bin/activate
# echo "find sample run at batch-output/training_run_${{SLURM_ARRAY_JOB_ID}}_1.out"

sleep $((RANDOM % 30 + 1))

python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-mechanisms-1/c2c7bezx        


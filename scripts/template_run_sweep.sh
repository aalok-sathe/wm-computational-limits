#!/bin/bash
#SBATCH --mail-user=aalok_sathe+wandb.ai@brown.edu
#SBATCH --mail-type=ALL

#SBATCH --mem-per-cpu 64G
# Request a GPU partition node and access to 1 GPU

#SBATCH -p 3090-gcondo --gres=gpu:1
#SBATCH -p gpu --gres=gpu:1 --account=carney-frankmj-condo

#SBATCH -a 1-10%15
#SBATCH -t 2-00:00:00

#SBATCH -o {batch_output_prefix}batch-output/training_run_%A_%a.out
#SBATCH --nodes=1 

set -x

. .venv/bin/activate
echo "find sample run at batch-output/training_run_${{SLURM_ARRAY_JOB_ID}}_1.out"

sleep $((RANDOM % 30 + 1))

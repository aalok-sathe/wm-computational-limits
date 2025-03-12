#!/bin/bash
#SBATCH --mail-user=aalok_sathe+wandb.ai@brown.edu
#SBATCH --mail-type=ALL

# Request a GPU partition node and access to 1 GPU
#SBATCH -p 3090-gcondo --gres=gpu:1
#SBATCH --mem-per-cpu 20G

#SBATCH -a 1-20%10
#SBATCH -t 2:00:00 ##shorter time because fewer training examples
#SBATCH -o batch-output/training_run_%A_%a.out


. .venv/bin/activate

echo "find sample run at batch-output/training_run_%A_1.out"



#Sweep 1 (You probably want to provide some documentation here so you remember what the sweep does without having to sift through the overview)
python3 -m workingmem  --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-0/6d4tifhz
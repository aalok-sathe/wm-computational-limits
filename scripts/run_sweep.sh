#!/bin/bash
#SBATCH --mail-user=aalok_sathe+wandb.ai@brown.edu
#SBATCH --mail-type=ALL

#SBATCH --mem-per-cpu 150G
# Request a GPU partition node and access to 1 GPU
#SBATCH -p 3090-gcondo --gres=gpu:1
#SBATCH -a 1-40%20
#SBATCH -t 1-00:00:00


##############################################
#### this set of parameters is for smaller models of d_model=128, which don't need as much memory or time to train
####        #SBATCH -p 3090-gcondo --gres=gpu:1
####        #SBATCH --mem-per-cpu 30G
####        #SBATCH -t 2:00:00
############################################## 
#### this set of parameters is for larger models which timed out on 4 hrs and OOM on 30G with a high dimensionality
####        #SBATCH -p 3090-gcondo --gres=gpu:1
####        #SBATCH --mem-per-cpu 150G
####        #SBATCH -t 5:00:00
##############################################
#### in case we want to use a different gpu condo at cpsy, we would pass in the following: 
####        #SBATCH -p l40s-gcondo --gres=gpu:1
##############################################

#SBATCH -o batch-output/training_run_%A_%a.out
#SBATCH --nodes=1 

set -x

. .venv/bin/activate
echo "find sample run at batch-output/training_run_${SLURM_ARRAY_JOB_ID}_1.out"


# we're going to try n_reg=100, concurrent_reg=2,4,8,16,32,64, seq_len=300, d_model=256
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-2/cbihadil # concurrent_reg=2
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-2/4tr728x0 # concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-2/l3x10vt6 # concurrent_reg=8
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-2/akw33l3m # concurrent_reg=16
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-2/lr79zu1i # concurrent_reg=32
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-2/iuh9nj9l # concurrent_reg=64


#### SWEEP for concurrent_reg = 64 with a bunch of different parameters including d_model, lr, weight decay, etc.

# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-3/t7y5f5jh # concurrent_reg=64

# we're going to try n_reg=100, concurrent_reg=2,4,8,16,32,64, seq_len=300, d_model=256 and SPLIT SET TRUE!
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-2/ # concurrent_reg=2
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-2/ # concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-2/ # concurrent_reg=8
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-2/ # concurrent_reg=16
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-2/ # concurrent_reg=32
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-2/ # concurrent_reg=64


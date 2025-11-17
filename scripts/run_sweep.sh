#!/bin/bash
#SBATCH --mail-user=aalok_sathe+wandb.ai@brown.edu
#SBATCH --mail-type=ALL

#SBATCH --mem-per-cpu 64G
# Request a GPU partition node and access to 1 GPU

#SBATCH -p 3090-gcondo --gres=gpu:1
##SBATCH -p gpu --gres=gpu:1 --account=carney-frankmj-condo
##SBATCH -p gpu-he --gres=gpu:1
##SBATCH -p l40s-gcondo --gres=gpu:1
##SBATCH -p cs-superlab-gcondo --gres=gpu:1 --account=cs-superlab-gcondo
##SBATCH -p gpu --gres=gpu:1 # no priority

#SBATCH -a 1-9%20
#SBATCH -t 2-00:00:00
##SBATCH -t 1-00:00:00


#SBATCH -o batch-output/training_run_%A_%a.out
#SBATCH --nodes=1 


set -x

. .venv/bin/activate
echo "find sample run at batch-output/training_run_${SLURM_ARRAY_JOB_ID}_1.out"

sleep $((RANDOM % 30 + 1))



# dataset.td_prob=0 dataset.role_n_congruence=0 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.1/be131vw7
# dataset.td_prob=0 dataset.role_n_congruence=0.25 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.1/m8c2cr7g
# dataset.td_prob=0 dataset.role_n_congruence=0.5 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.1/lezq9cbe
# dataset.td_prob=0 dataset.role_n_congruence=0.75 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.1/8g76x2z1
# dataset.td_prob=0 dataset.role_n_congruence=1 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.1/pedc15j6

# dataset.td_prob=1 dataset.role_n_congruence=0 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.1/crjufe02
# dataset.td_prob=1 dataset.role_n_congruence=0.25 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.1/lpgcp60v
# dataset.td_prob=1 dataset.role_n_congruence=0.5 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.1/hzflaqxx
# dataset.td_prob=1 dataset.role_n_congruence=0.75 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.1/ck9pi0go
# dataset.td_prob=1 dataset.role_n_congruence=1 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.1/w03izn1c
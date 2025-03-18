#!/bin/bash
#SBATCH --mail-user=aalok_sathe+wandb.ai@brown.edu
#SBATCH --mail-type=ALL

# Request a GPU partition node and access to 1 GPU
#SBATCH -p 3090-gcondo --gres=gpu:1
#SBATCH --mem-per-cpu 20G

#SBATCH -a 1-29%50
#SBATCH -t 3:00:00 ##shorter time (2hrs) because fewer training examples if 10_000, but make it 3hrs for 100_000
#SBATCH -o batch-output/training_run_%A_%a.out


. .venv/bin/activate

echo "find sample run at batch-output/training_run_%A_1.out"



#Sweep 1 (You probably want to provide some documentation here so you remember what the sweep does without having to sift through the overview)
# 100 runs, n_reg 100 
# python3 -m workingmem  --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-0/kklynomy

# Sweep 2:  n_reg 2, concurrent 2, 
# python3 -m workingmem  --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-0/5kingyzz

# Sweep 3: n_reg 2, concurrent 2, concurrent items 3, n_items 50, lr 1e-3
# python3 -m workingmem  --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-0/2vqsw23g

# Sweep 4: n_reg 2, concurrent 2, concurrent items 3, n_items 50, lr 1e-3, n_train 100_000
# python3 -m workingmem  --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-0/3o8n61va

# Sweep 5: n_reg 2/5, concurrent 2, concurrent items 3, n_items 20, lr 1e-2:1e-5, wt decay 0:1e-4, n_heads 2/4, d_model 64/128/256, batchsize 128, epochs 100, n_train 100_000
# python3 -m workingmem  --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-0/mjzbrq26


# Sweep 6: n_reg 2/5, concurrent 2, concurrent items 3, n_items 50, lr 1e-4, wt decay 3e-4, n_heads 2, d_model 128, batchsize 128, epochs 600, n_train 100_000, seq_len 10
# python3 -m workingmem  --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-0/rh12jasn
# python3 -m workingmem  --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-0/4wszpts4

########################################
#### Dataset generation code fixed! #### 
#### commit 6596a05
########################################

# Sweep 7: ignore_prob .5, seq_len 10, all else same. n_reg 2/5
# python3 -m workingmem  --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-0/hjalz9la

# Sweep 8: n_reg 50, concurrent_reg 2/3, seq_len 50
# python3 -m workingmem  --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-0/iyl8lx6x

# Sweep 9 same but with n_train 10_000 instead of 100_000
# python3 -m workingmem  --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-0/69f0hgsl

# Sweep 10a: n_reg 3, concurrent_reg 3, seq_len 10, n_train 100_000 (to try to match the conditions of Aneri/Aaron's experiments)
# python3 -m workingmem  --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-0/byffj3o2
# Sweep 10b: n_reg 2, concurrent_reg 2, seq_len 10, n_train 100_000 (to try to match the conditions of Aneri/Aaron's experiments)
python3 -m workingmem  --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-0/
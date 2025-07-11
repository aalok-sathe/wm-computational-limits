#!/bin/bash
#SBATCH --mail-user=aalok_sathe+wandb.ai@brown.edu
#SBATCH --mail-type=ALL

# Request a GPU partition node and access to 1 GPU
#SBATCH -p 3090-gcondo --gres=gpu:1
#####SBATCH -p l40s-gcondo --gres=gpu:1
#SBATCH --mem-per-cpu 30G

#SBATCH -a 1-20%20
#SBATCH -t 1:50:00 ##shorter time (1hrs) because fewer training examples if 10_000, but make it 2hrs for 100_000
####SBATCH -t 6:00:00 ##shorter time (1hrs) because fewer training examples if 10_000, but make it 2hrs for 100_000
#SBATCH -o batch-output/training_run_%A_%a.out
#SBATCH --nodes=1 

set -x

. .venv/bin/activate

echo "find sample run at batch-output/training_run_%A_1.out"



# Sweep 1 (You probably want to provide some documentation here so you remember what the sweep does without having to sift through the overview)
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
#### Dataset generation code fixed!    # 
#### commit 6596a05                    #
########################################

# Sweep 7: ignore_prob .5, seq_len 10, all else same. n_reg 2/5
# python3 -m workingmem  --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-0/hjalz9la

# Sweep 8: n_reg 50, concurrent_reg 2/3, seq_len 50
# python3 -m workingmem  --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-0/iyl8lx6x

# Sweep 9 same but with n_train 10_000 instead of 100_000
# python3 -m workingmem  --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-0/69f0hgsl

# Sweep 10a: n_reg 3, concurrent_reg 3, seq_len 10, n_train 100_000 (to try to match the conditions of Aneri/Aaron's experiments)
# python3 -m workingmem  --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-0/byffj3o2

##########################################
#### Dataset generation code amended!    # 
#### commit c7a84fd                      #
##########################################

# Sweep 11a: n_reg 3, concurrent_reg 3, seq_len 10, n_train 100_000 (to try to match the conditions of Aneri/Aaron's experiments)
# python3 -m workingmem  --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-0/x3ef5jj1
# Sweep 11b: n_reg 2, concurrent_reg 2, seq_len 10, n_train 100_000 (to try to match the conditions of Aneri/Aaron's experiments)
# python3 -m workingmem  --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-0/84m62mn1
# Sweep 11c: n_reg 3, concurrent_reg 2, seq_len 10, n_train 100_000 (to try to match the conditions of Aneri/Aaron's experiments EXCEPT make n_reg=3)
# python3 -m workingmem  --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-0/zctyznwg

# Sweep 12: a repeat of Sweep 11a-c but with n_train 10_000
# python3 -m workingmem  --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-0/rb7u44gj # n_reg 2 concurrent 2
# python3 -m workingmem  --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-0/gv8oc4n3 # n_reg 3 concurrent 2
# python3 -m workingmem  --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-0/7dbv1sd9 # n_reg 3 concurrent 3

# Sweep 13: n_train 100_000, n_reg 50, seq_len 20
# python3 -m workingmem  --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-0/shwv3iys # concurrent_reg 2
# python3 -m workingmem  --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-0/8qejrp9n # concurrent_reg 3
# python3 -m workingmem  --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-0/x0wla7ce # concurrent_reg 4


# this is a repeat of sweeps 11a-c but with more granular logging as of commit:4f5dc7c
# Sweep 11a: n_reg 3, concurrent_reg 3, seq_len 10, n_train 100_000 (to try to match the conditions of Aneri/Aaron's experiments)
# python3 -m workingmem  --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-0/tdyap6gh
# Sweep 11b: n_reg 2, concurrent_reg 2, seq_len 10, n_train 100_000 (to try to match the conditions of Aneri/Aaron's experiments)
# python3 -m workingmem  --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-0/iswucmve
# Sweep 11c: n_reg 3, concurrent_reg 2, seq_len 10, n_train 100_000 (to try to match the conditions of Aneri/Aaron's experiments EXCEPT make n_reg=3)
# python3 -m workingmem  --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-0/wcdpxsu2

##########################################
# changes to checkpointing and to some   #
# crucial dataset and model params       #
# (seq_len=14); (d_model=32)             #
#### commits e05464-89c05b               #
##########################################

# Sweep 12a: n_reg 2, concurrent 2
# python3 -m workingmem  --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-0/vc7i09gs
# Sweep 12b: n_reg 3, concurrent 3
# python3 -m workingmem  --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-0/h3kgrek1
# Sweep 12c: n_reg 50, concurrent 1 (for symbol pretraining)
# python3 -m workingmem  --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-0/evcxg3kc

# Sweep 13a n_reg 2, concurrent 2 with 8-dim embeddings
# python3 -m workingmem  --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-0/o7fmq18y
# Sweep 13b: n_reg 3, concurrent 3 with 8-dim embeddings
# python3 -m workingmem  --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-0/gsbkbheh 

#### FROZEN EMBEDDINGS!
# Sweep 14a FROM PRETRAINED [evcxg3kc]! n_reg 2, concurrent 2 with 32-dim embeddings, 40 epochs
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-0/xs7tnwgc #--array_task_id "$SLURM_ARRAY_TASK_ID"
# Sweep 14b FROM PRETRAINED [evcxg3kc]! n_reg 3, concurrent 3 with 32-dim embeddings, 40 epochs
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-0/uzrvtykz #--array_task_id "$SLURM_ARRAY_TASK_ID"
# Sweep 14c FROM PRETRAINED [vc7i09gs]! n_reg 3, concurrent 3 with 32-dim embeddings, 40 epochs CANCELLED
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-0/qryq9z1q 

#### WARM EMBEDDINGS!
# Sweep 14a FROM PRETRAINED [evcxg3kc]! n_reg 2, concurrent 2 with 32-dim embeddings, 40 epochs
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-0/gp9hkgwc
# Sweep 14b FROM PRETRAINED [evcxg3kc]! n_reg 3, concurrent 3 with 32-dim embeddings, 40 epochs
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-0/ce8lnv4r
# Sweep 14c FROM PRETRAINED [vc7i09gs]! n_reg 3, concurrent 3 with 32-dim embeddings, 40 epochs
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-0/72e6vqq7 


##########################################
#### wm-comp-limits-1                 ####
##########################################
# Sweep 1: n_reg 5, concurrent 1--5, seq_len 14, n_train 100_000
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-1/1e53k4tn # concurrent 1
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-1/1ohh210b # concurrent 2
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-1/olt8ye2v # concurrent 3
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-1/d6lo7sid # concurrent 4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-1/2h3s5i1j # concurrent 5 [5,5]
#

# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-1/8l957ug3 # 3,3
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-1/aacwus5y # 2,2
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-1/hez1hokq # 4,4

# Sweep 2: pre-training for split-set control
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-1/iiu16j21 # split set true
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-1/c2hoksay # split set false
# now, we fine-tune on 3, 3 with these pretraining conditions 
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-1/rmc4grdu # split set true, concurrent 3, n_reg 3
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-1/m9rs053w # split set false, concurrent 3, n_reg 3

# Sweep R: exactly replicate the split set stuff but with 128-dim
# PRETRAIN
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-1/nxgusfzl # split set true
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-1/b931g4g8 # split set false


# FINETUNE 
#somehow seq_len was 25???
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-1/h0x7w7g3 # split set false 3,3 # this is a comparison vanilla training run
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-1/677uqtug # pretrained on split set false 3,3
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-1/blyrm120 # pretrained on split set false 3,3
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-1/7qdu9mqj # pretrained on split set true 2,2
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-1/edun0q37 # pretrained on split set true 3,3

# FINETUNE 
# seq len 14
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-1/y81xhmif # pretrained on split set true 2,2
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-1/y77bap5n # pretrained on split set false 2,2

# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-1/r02l0m2c # pretrained on split set true 3,3
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-1/e1n77dxd # pretrained on split set false 3,3


##########
# we're going to try n_reg=10, concurrent_reg=2..6, seq_len=35
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-2/roodafij # concurrent_reg=2
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-2/hb5q5efm # concurrent_reg=3
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-2/cxnz9utf # concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-2/dyh8frn6 # concurrent_reg=5
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-2/7w6njh78 # concurrent_reg=6

##########
# n_reg=50, concurrent_reg=2, seq_len=35
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-2/9q1up642 # split_set false
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-2/cboc5to0 # split_set true

# we're going to try n_reg=100, concurrent_reg=2,4,8,16,32,64, seq_len=300, d_model=256
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-2/cbihadil # concurrent_reg=2
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-2/4tr728x0 # concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-2/l3x10vt6 # concurrent_reg=8
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-2/akw33l3m # concurrent_reg=16
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-2/lr79zu1i # concurrent_reg=32
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-2/iuh9nj9l # concurrent_reg=64


# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-3/ # concurrent_reg=64
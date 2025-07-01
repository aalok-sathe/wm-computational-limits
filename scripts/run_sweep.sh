#!/bin/bash
#SBATCH --mail-user=aalok_sathe+wandb.ai@brown.edu
#SBATCH --mail-type=ALL

#SBATCH --mem-per-cpu 50G
# Request a GPU partition node and access to 1 GPU
#SBATCH -p 3090-gcondo --gres=gpu:1
##SBATCH -p carney-gcondo --gres=gpu:1
##SBATCH -p gpu-he --gres=gpu:1
##SBATCH -p l40s-gcondo --gres=gpu:1
##SBATCH -p cs-superlab-gcondo --gres=gpu:1 --account=cs-superlab-gcondo

#SBATCH -a 1-1%20
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
####        #SBATCH -t 1-00:00:00
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
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-3/ # concurrent_reg=2
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-3/2qv55d5j # concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-3/7xpzajt3 # concurrent_reg=8
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-3/22pwqoqu # concurrent_reg=16
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-3/rjm7c9an # concurrent_reg=32
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-3/ghl93z95 # concurrent_reg=64


#### SWEEP for concurrent_reg = 64 with a bunch of different parameters including d_model, lr, weight decay, etc.

# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-3/bz8gqvob # concurrent_reg=32
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-3/lntph15o # concurrent_reg=64

# what changed here?
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-3/m16lpz4x # concurrent_reg=64
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-3/7ttzwpnh # concurrent_reg=32 
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-3/9ahxksw1 # concurrent_reg=16

# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-3/kckcjcyy # concurrent_reg=2
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-3/8gae8alk # concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-3/ots5buxo # concurrent_reg=8


#DISCARDED fix some params after running the hparam sweep
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-3/kd5misrb # concurrent_reg=64
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-3/ws13o297 # concurrent_reg=32
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-3/q1nchbeb # concurrent_reg=16
#END DISCARDED 

# grid here on with 20 seeds with some fixed parameters

# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-3/y3rygm69 # concurrent_reg=64
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-3/o7pe2om2 # concurrent_reg=32
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-3/0jtss6xy # concurrent_reg=16

# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-3/66m9eheb # concurrent_reg=8
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-3/kus0bfuk # concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-3/bz0vzcwc # concurrent_reg=2


# this next part is for a 20-grid-search following the random sweeps performed for each condition separately
#
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-4/zif3avh1 # concurrent_reg=64
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-4/7k8f2oe4 # concurrent_reg=32
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-4/ahv9p59l # concurrent_reg=16
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-4/stkof2md # concurrent_reg=8
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-4/rf8k5pvu # concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-4/qc820c8e # concurrent_reg=2
#
####

# in this section, we are experimenting with post-training after already having learned a strategy on 100_2 task.
# we will, however, do all the conditions with the same architecture as 100_2 so that weights can be initialized.

# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-4/fpwt03ja # concurrent_reg=64
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-4/vg3ppbzw # concurrent_reg=32
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-4/jfvyqc4m # concurrent_reg=16
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-4/8msre3gp # concurrent_reg=08
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-4/0xe1g0uu # concurrent_reg=04

# below, we will run matched split-set control versions of the task for each of 2,4,8,16,32,64 concurrent_reg
# with hyperparams prevoiusly found in per-condition hyperparam sweeps

# sleep a random amount of time to avoid all jobs starting at the same time between 1 to 30 seconds---this is so
# each run has time to query wandb and pick a different seed (though I'd have imagined wandb to take care of this
# already)
sleep $((RANDOM % 30 + 1))

# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-4/qn4l79ft # concurrent_reg=64
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-4/lddb039a # concurrent_reg=32
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-4/1skc9gap # concurrent_reg=16
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-4/vgoktqah # concurrent_reg=08
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-4/ur3f03fm # concurrent_reg=04
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-4/tc326x3w # concurrent_reg=02


# !!!⚠
# REDO OF THE best-shot SWEEPS this time with just 4 concurrent items (the 128 concurrent items condition
# introduces a quasi split-set condition, enabling heuristic solutions unless really pressured to do the task
# in higher register conditions)
# this next part is for a 20-grid-search following the random sweeps performed for each condition separately
#
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-5/ # concurrent_reg=64
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-5/ # concurrent_reg=32
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-5/ # concurrent_reg=16
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-5/ # concurrent_reg=8
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-5/ # concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-5/ # concurrent_reg=2
#
####
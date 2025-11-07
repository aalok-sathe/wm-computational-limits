#!/bin/bash
#SBATCH --mail-user=aalok_sathe+wandb.ai@brown.edu
#SBATCH --mail-type=ALL

#SBATCH --mem-per-cpu 64G
# Request a GPU partition node and access to 1 GPU

##SBATCH -p 3090-gcondo --gres=gpu:1
#SBATCH -p gpu --gres=gpu:1 --account=carney-frankmj-condo
##SBATCH -p gpu-he --gres=gpu:1
##SBATCH -p l40s-gcondo --gres=gpu:1
##SBATCH -p cs-superlab-gcondo --gres=gpu:1 --account=cs-superlab-gcondo
##SBATCH -p gpu --gres=gpu:1 # no priority

#SBATCH -a 1-9%20
#SBATCH -t 2-00:00:00
##SBATCH -t 1-00:00:00


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

sleep $((RANDOM % 30 + 1))

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
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-5/12yogz48 # concurrent_reg=64
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-5/bqbrw2fo # concurrent_reg=32
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-5/nq7pm7rd # concurrent_reg=16
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-5/puepb2hf # concurrent_reg=8
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-5/ekpvhazy # concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-5/rjgsml9b # concurrent_reg=2
#
####

##### having done the above sweeps, we're going to fix hparams and run 15 random seeds for each condition
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-6/w1wz6rb4 # concurrent_reg=64
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-6/1uo37hjr # concurrent_reg=32
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-6/dh52ts01 # concurrent_reg=16
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-6/c9q2afm7 # concurrent_reg=8
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-6/i0kefy4p # concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-6/qc4uk70r # concurrent_reg=2
#
# running a new sweep for 8 reg with the hparams for 32 above
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-6/m6wg25v9 # concurrent_reg=8

# testing sparsity implementation
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-6.6/kw810xdk # concurrent_reg 2
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-6.6.1/7hfu7len # concurrent_reg 2 # rescale loss, not epochs 
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-6.7/j3wx7mlr # concurrent_reg 2 # rescale loss, not epochs 

# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-6.7/0l3s4hrp # concurrent_reg=2 NO MASK ANS TOKENS


# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-6.8/t64sacz1 # concurrent_reg=2



#  a. why were some of the sparsity runs tanking? this is unexpected behavior---they should hover around chance accuracy if they don't learn
#  b. ^ to diagnose (a), if we didn't pick a large enough range of hparams, re-run with a larger set of hparams (esp. learning rates, since the previous sweep seemed to bottom out at a not-high-enough learning rate)
#  c. using ^ b, fix hparams and run two identical sweeps that vary only in:
#  mask_answer_tokens and sparsity to be able to make a comparison between
#  masking and non masking, for sparsity levels 0 and 0.2. possible outcomes: (1)
#  no difference in mask vs no mask conditions for sparsity=0 [models generally
#  solve the task the same way in both cases]. (2) some difference in mask vs no
#  mask conditions for sparsity=0 [models, when presented with answer tokens, use
#  them as part of its strategy, making the problem easier?]

# c.
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-6.9/n94oktf6 # concurrent_reg=2; mask ans tokens=1
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-6.9/otb62v8r # concurrent_reg=2; mask ans tokens=0


# here, we are doing a hparam sweep for 3-roles
# first, sparsity=0
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7/vu2givgu # concurrent_reg=3; mask ans tokens=1
# next, sparsity=0.2
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7/satdeohl # concurrent_reg=3; mask ans tokens=1

# here, we are doing a hparam sweep for sparsity=0.4 [tabled for now]
# roles=2
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7/ui1g0lbp # concurrent_reg=2; mask ans tokens=1
# roles=3
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7/njn1x9pp # concurrent_reg=3; mask ans tokens=1


# roles=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7/33rfjd2l # concurrent_reg=4; sparsity=0.4; mask ans tokens=1
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7/07e9bazb # concurrent_reg=4; sparsity=0.2; mask ans tokens=1
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7/5jmzfirn # concurrent_reg=4; sparsity=0.0; mask ans tokens=1

# roles=8
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7/1pyb857k # concurrent_reg=8; sparsity=0.4; mask ans tokens=1
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7/swytfjrq # concurrent_reg=8; sparsity=0.2; mask ans tokens=1
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7/xlohfyqc # concurrent_reg=8; sparsity=0.0; mask ans tokens=1

# FIXED hparams
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7/p37rv7lo # concurrent_reg=3; sparsity=0.2; mask ans tokens=1
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7/he3rvni1 # concurrent_reg=3; sparsity=0.0; mask ans tokens=1
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7/7oflvb9a # concurrent_reg=3; sparsity=0.4; mask ans tokens=1
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7/mc3o93qc # concurrent_reg=2; sparsity=0.4; mask ans tokens=1

# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7/mtkrgqkq # concurrent_reg=4; sparsity=0.0; mask ans tokens=1
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7/h2919rom # concurrent_reg=4; sparsity=0.2; mask ans tokens=1
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7/ymioopaw # concurrent_reg=4; sparsity=0.4; mask ans tokens=1

# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7/dwgvihqr # concurrent_reg=8; sparsity=0.0; mask ans tokens=1
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7/1t8u5m81 # concurrent_reg=8; sparsity=0.2; mask ans tokens=1
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7/gsb7upw8 # concurrent_reg=8; sparsity=0.4; mask ans tokens=1



# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.1.0/4kyhxuh7 # concurrent_reg=8; sparsity=0.0; challenge=1 # debug: n_train = 1000
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.1.0/k5bbaray # concurrent_reg=8; sparsity=0.0; challenge=1
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.1.0/6qqcxtzt # concurrent_reg=4; sparsity=0.0; challenge=1 # OOM??
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.1.0/4n4p3cp7 # concurrent_reg=4; sparsity=0.0; challenge=1 # OOM??

# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.1.0/helwa8d8 # concurrent_reg=2; sparsity=0.0; challenge=1 heldout=15/50
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.1.0/ckjfjz9z # concurrent_reg=2; sparsity=0.0; challenge=1 heldout=30/50

# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.1.0/gpdnruyk # concurrent_reg=2 [n_reg=2]; sparsity=0.0; challenge=1 heldout=30/50
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.1.0/4il3ey75 # concurrent_reg=2 [n_reg=10]; sparsity=0.0; challenge=1 heldout=30/50 
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.1.0/qp9ih3m9 # concurrent_reg=3 [n_reg=10]; sparsity=0.0; challenge=1 heldout=30/50 
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.1.0/vhay2jri # concurrent_reg=2 [n_reg=3]; sparsity=0.0; challenge=1 heldout=30/50 


###################
# in this next section we're going to re-run the set of sparsity experiments but with a held-out challenge-set of items per register
# [n_reg=100];

# CONCURRENT=2
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.2.1/prygyov4 # sparsity=0.0; heldout=15/50 
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.2.1/gh3zw5mw # sparsity=0.2; heldout=15/50 
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.2.1/vg8d7uq1 # sparsity=0.4; heldout=15/50 

# CONCURRENT=3
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.2.1/goo9782t # sparsity=0.0; heldout=15/50 
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.2.1/lrcvrp3y # sparsity=0.2; heldout=15/50 
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.2.1/qenlwm1s # sparsity=0.4; heldout=15/50 

# CONCURRENT=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.2.1/zyutnr5j # sparsity=0.0; heldout=15/50 
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.2.1/af29npcn # sparsity=0.2; heldout=15/50 
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.2.1/alt0vj30 # sparsity=0.4; heldout=15/50 

# CONCURRENT=8
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.2.1/h5nzbv6p # sparsity=0.0; heldout=15/50 
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.2.1/7rjd9kga # sparsity=0.2; heldout=15/50 
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.2.1/42w47r0l # sparsity=0.4; heldout=15/50 


##########
# we're going to do a bunch of sweeps over conditions keeping concurrent and sparsity values fixed
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.2.2/7rethi6d # conc=2 sparsity=0.0; heldout=15/50 
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.2.2/76td38tj # conc=2 sparsity=0.2; heldout=15/50 
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.2.2/k124yh1o # conc=2 sparsity=0.4; heldout=15/50 

# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.2.2/8bq1fdgp # conc=4 sparsity=0.0; heldout=15/50 
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.2.2/tef8fbh3 # conc=4 sparsity=0.2; heldout=15/50 
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.2.2/e9ubytqx # conc=4 sparsity=0.4; heldout=15/50 

# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.2.2/noan0dcc # conc=8 sparsity=0.0; heldout=15/50 
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.2.2/ojj1rtet # conc=8 sparsity=0.2; heldout=15/50 
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.2.2/9ap2ww8j # conc=8 sparsity=0.4; heldout=15/50 


# td prob 0,.5,1 n_back 5
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.0/gncj6lbu # conc=2 sparsity=0.0; heldout=15/50 
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.0/j3nhalg5 # conc=4 sparsity=0.0; heldout=15/50 
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.0/gydtzx1k # conc=8 sparsity=0.0; heldout=15/50 


# dataset.td_prob=0
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.1/wuj0loqv
# dataset.td_prob=0.5
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.1/wloq2dzr
# dataset.td_prob=0.7
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.1/8rk9505v
# dataset.td_prob=0.8
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.1/13ijgrap
# dataset.td_prob=0.9
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.1/4rrckcd1
# dataset.td_prob=1
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.1/0edk808h
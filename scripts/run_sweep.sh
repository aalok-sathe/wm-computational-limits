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

#SBATCH -a 1-9%15
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

# -----
# dataset.td_prob=0 dataset.role_n_congruence=0.5 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.1/xgl4bz2v
# -----
# dataset.td_prob=0 dataset.role_n_congruence=0.6 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.1/8cn96ij0
# dataset.td_prob=0 dataset.role_n_congruence=0.7 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.1/2y7n1di3
# dataset.td_prob=0 dataset.role_n_congruence=0.8 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.1/wj414e29
# dataset.td_prob=0 dataset.role_n_congruence=0.9 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.1/hzfdqof0
# -----
# dataset.td_prob=0 dataset.role_n_congruence=1 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.1/7xf2lt09


# ------ NO POSITIONAL EMBEDDINGS!

# dataset.td_prob=0 dataset.role_n_congruence=0 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.2/u83o3qw7
# dataset.td_prob=0 dataset.role_n_congruence=0.25 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.2/7tmqmeu2
# dataset.td_prob=0 dataset.role_n_congruence=0.5 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.2/967wqail
# dataset.td_prob=0 dataset.role_n_congruence=0.7 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.2/qvelzjzq
# dataset.td_prob=0 dataset.role_n_congruence=0.8 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.2/0sh645ly
# dataset.td_prob=0 dataset.role_n_congruence=0.9 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.2/g8gux4um
# dataset.td_prob=0 dataset.role_n_congruence=0.95 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.2/11ibquvt
# dataset.td_prob=0 dataset.role_n_congruence=1 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.2/dxuufdrs

# dataset.td_prob=1 dataset.role_n_congruence=0 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.2/4nn7p2va
# dataset.td_prob=1 dataset.role_n_congruence=0.25 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.2/h3gppajd
# dataset.td_prob=1 dataset.role_n_congruence=0.5 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.2/4anfot6j
# dataset.td_prob=1 dataset.role_n_congruence=0.7 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.2/y9024pym
# dataset.td_prob=1 dataset.role_n_congruence=0.8 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.2/jkejkuvi
# dataset.td_prob=1 dataset.role_n_congruence=0.9 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.2/gp1k1ob7
# dataset.td_prob=1 dataset.role_n_congruence=0.95 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.2/2dnxagjz
# dataset.td_prob=1 dataset.role_n_congruence=1 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.2/hu3vslvz


# ------------ hparam sweep for NoPE
# dataset.td_prob=0 dataset.role_n_congruence=0 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.2a0/0uayie2m
# dataset.td_prob=1 dataset.role_n_congruence=0 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.2a0/veme6eqt

# ------------ hparam sweep for NoPE but with `dataset.concurrent_reg=2`
# dataset.td_prob=0 dataset.role_n_congruence=0 dataset.n_back=2 dataset.concurrent_reg=2 dataset.seq_len=100
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.2a0/aqznrnfg
# dataset.td_prob=1 dataset.role_n_congruence=0 dataset.n_back=2 dataset.concurrent_reg=2 dataset.seq_len=100
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.2a0/z3d33pg6

# ------------ hparam sweep for 'rotary' with ignore-aware N-back and Role-N cong
# dataset.td_prob=0 dataset.role_n_congruence=0 dataset.ignore_prob=0.5 dataset.n_back=4 dataset.concurrent_reg=4 dataset.seq_len=200
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.2a1/jd3hu5uz
# dataset.td_prob=1 dataset.role_n_congruence=0 dataset.ignore_prob=0.5 dataset.n_back=4 dataset.concurrent_reg=4 dataset.seq_len=200
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.2a1/jdcdu1yi


# ------------
# dataset.td_prob=0 dataset.role_n_congruence=0 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.2b1/sorf4kwc
# dataset.td_prob=0 dataset.role_n_congruence=0.3 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.2b1/02k8w5iz
# dataset.td_prob=0 dataset.role_n_congruence=0.5 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.2b1/8ei8xdx1
# dataset.td_prob=0 dataset.role_n_congruence=0.7 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.2b1/ypop39i9
# dataset.td_prob=0 dataset.role_n_congruence=0.8 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.2b1/izx3dog9
# dataset.td_prob=0 dataset.role_n_congruence=0.9 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.2b1/5z7epshg
# dataset.td_prob=0 dataset.role_n_congruence=0.95 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.2b1/j15lk3ai
# dataset.td_prob=0 dataset.role_n_congruence=1 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.2b1/ue6iof7n

# dataset.td_prob=1 dataset.role_n_congruence=0 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.2b1/g2zkgrsd
# dataset.td_prob=1 dataset.role_n_congruence=0.3 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.2b1/tak7mn7x
# dataset.td_prob=1 dataset.role_n_congruence=0.5 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.2b1/qvd0rk76
# dataset.td_prob=1 dataset.role_n_congruence=0.7 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.2b1/7anrebwz
# dataset.td_prob=1 dataset.role_n_congruence=0.8 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.2b1/ygek6txx
# dataset.td_prob=1 dataset.role_n_congruence=0.9 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.2b1/edvv9qx5
# dataset.td_prob=1 dataset.role_n_congruence=0.95 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.2b1/aa7uyscq
# dataset.td_prob=1 dataset.role_n_congruence=1 dataset.n_back=4 dataset.concurrent_reg=4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.2b1/3rstk75h


# finetuning experiments concurrent/n-back=4 --> 4
# dataset.td_prob=0 dataset.role_n_congruence=0 dataset.n_back=4 dataset.concurrent_reg=4 model.from_pretrained=model_checkpoints/sorf4kwc
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.2c0/f6qupz58
# dataset.td_prob=0 dataset.role_n_congruence=0 dataset.n_back=4 dataset.concurrent_reg=4 model.from_pretrained=model_checkpoints/g2zkgrsd
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.2c0/e2vekx5n
# dataset.td_prob=1 dataset.role_n_congruence=0 dataset.n_back=4 dataset.concurrent_reg=4 model.from_pretrained=model_checkpoints/sorf4kwc
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.2c0/s460b0wx
# dataset.td_prob=1 dataset.role_n_congruence=0 dataset.n_back=4 dataset.concurrent_reg=4 model.from_pretrained=model_checkpoints/g2zkgrsd
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.2c0/heesdnzg


# finetuning experiments concurrent/n-back=4 --> 5
# dataset.td_prob=0 dataset.role_n_congruence=0 dataset.n_back=5 dataset.concurrent_reg=5 model.from_pretrained=model_checkpoints/sorf4kwc
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.2c0/dyvm5w5c
# dataset.td_prob=0 dataset.role_n_congruence=0 dataset.n_back=5 dataset.concurrent_reg=5 model.from_pretrained=model_checkpoints/g2zkgrsd
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.2c0/92kwfebg
# dataset.td_prob=1 dataset.role_n_congruence=0 dataset.n_back=5 dataset.concurrent_reg=5 model.from_pretrained=model_checkpoints/sorf4kwc
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.2c0/fm6sdl37
# dataset.td_prob=1 dataset.role_n_congruence=0 dataset.n_back=5 dataset.concurrent_reg=5 model.from_pretrained=model_checkpoints/g2zkgrsd
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.3.2c0/21xm4gb2
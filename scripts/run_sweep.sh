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

#SBATCH -a 1-10%15
#SBATCH -t 2-00:00:00
##SBATCH -t 1-00:00:00


#SBATCH -o batch-output/training_run_%A_%a.out
#SBATCH --nodes=1 


set -x

. .venv/bin/activate
echo "find sample run at batch-output/training_run_${SLURM_ARRAY_JOB_ID}_1.out"

sleep $((RANDOM % 30 + 1))




# dataset.td_prob=0 dataset.role_n_congruence=0 dataset.ignore_prob=0.5 dataset.n_back=4 dataset.concurrent_reg=4 dataset.seq_len=200 model.model_class=rnn
# Sweep URL: https://wandb.ai/aloxatel/wm-comp-limit-7.4.0c2/sweeps/wys7ypvd
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.0c2/wys7ypvd

# dataset.td_prob=0 dataset.role_n_congruence=0 dataset.ignore_prob=0.5 dataset.n_back=4 dataset.concurrent_reg=4 dataset.seq_len=200 model.model_class=lstm
# Sweep URL: https://wandb.ai/aloxatel/wm-comp-limit-7.4.0c2/sweeps/oz2910d4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.0c2/oz2910d4


# ---- -------- new sweep ----
# wandb: WARNING Malformed sweep config detected! This may cause your sweep to behave in unexpected ways.
# wandb: WARNING To avoid this, please fix the sweep config schema violations below:
# wandb: WARNING   Violation 1. Additional properties are not allowed ('create_sweep', 'from_config', 'project_name', 'run_sweep', 'sweep_id' were unexpected)
# Create sweep with ID: wbfar9j0
# Sweep URL: https://wandb.ai/aloxatel/wm-comp-limit-7.4.0c2/sweeps/wbfar9j0
# dataset.td_prob=1 dataset.role_n_congruence=0 dataset.ignore_prob=0.5 dataset.n_back=4 dataset.concurrent_reg=4 dataset.seq_len=200 model.model_class=rnn
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.0c2/wbfar9j0

# ---- -------- new sweep ----
# wandb: WARNING Malformed sweep config detected! This may cause your sweep to behave in unexpected ways.
# wandb: WARNING To avoid this, please fix the sweep config schema violations below:
# wandb: WARNING   Violation 1. Additional properties are not allowed ('create_sweep', 'from_config', 'project_name', 'run_sweep', 'sweep_id' were unexpected)
# Create sweep with ID: 1z2tey19
# Sweep URL: https://wandb.ai/aloxatel/wm-comp-limit-7.4.0c2/sweeps/1z2tey19
# dataset.td_prob=1 dataset.role_n_congruence=0 dataset.ignore_prob=0.5 dataset.n_back=4 dataset.concurrent_reg=4 dataset.seq_len=200 model.model_class=lstm
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.0c2/1z2tey19

# pretraining on 4-role and 4-back task to then test near- and far-transfer
# NBACK
# rnn
# dataset.td_prob=1 dataset.role_n_congruence=0 dataset.ignore_prob=0.5 dataset.n_back=4 dataset.concurrent_reg=4 dataset.seq_len=200 model.model_class=rnn model.d_hidden=512 model.d_model=512 model.n_layers=2
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.1c0/cjtwcbfr
# lstm
# dataset.td_prob=1 dataset.role_n_congruence=0 dataset.ignore_prob=0.5 dataset.n_back=4 dataset.concurrent_reg=4 dataset.seq_len=200 model.model_class=lstm model.d_hidden=512 model.d_model=512 model.n_layers=2
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.1c0/bk1zoetn

# REFBACK
# rnn
# dataset.td_prob=0 dataset.role_n_congruence=0 dataset.ignore_prob=0.5 dataset.n_back=4 dataset.concurrent_reg=4 dataset.seq_len=200 model.model_class=rnn model.d_hidden=512 model.d_model=512 model.n_layers=2
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.1c0/70rfkjiv
# lstm
# dataset.td_prob=0 dataset.role_n_congruence=0 dataset.ignore_prob=0.5 dataset.n_back=4 dataset.concurrent_reg=4 dataset.seq_len=200 model.model_class=lstm model.d_hidden=512 model.d_model=512 model.n_layers=2
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.1c0/t4wviuxs

# FINETUNING on 5-role and 5-back task to then test near- and far-transfer

# [deleted]

#### UGH all the above finetuning experiments somehow used transformer as model class
# dataset.td_prob=1 dataset.role_n_congruence=0 dataset.n_back=5 dataset.concurrent_reg=5 model.model_class=rnn trainer.learning_rate=2e-4 model.from_pretrained=model_checkpoints/70rfkjiv
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.1c1_pt/j670n5w3
# dataset.td_prob=1 dataset.role_n_congruence=0 dataset.n_back=5 dataset.concurrent_reg=5 model.model_class=rnn trainer.learning_rate=2e-4 model.from_pretrained=model_checkpoints/cjtwcbfr
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.1c1_pt/79pspsue

# dataset.td_prob=0 dataset.role_n_congruence=0 dataset.n_back=5 dataset.concurrent_reg=5 model.model_class=rnn trainer.learning_rate=2e-4 model.from_pretrained=model_checkpoints/70rfkjiv
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.1c1_pt/ju88er93
# dataset.td_prob=0 dataset.role_n_congruence=0 dataset.n_back=5 dataset.concurrent_reg=5 model.model_class=rnn trainer.learning_rate=2e-4 model.from_pretrained=model_checkpoints/cjtwcbfr
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.1c1_pt/oy3pz83r

# dataset.td_prob=0 dataset.role_n_congruence=0 dataset.n_back=5 dataset.concurrent_reg=5 model.model_class=lstm trainer.learning_rate=1e-3 model.from_pretrained=model_checkpoints/t4wviuxs
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.1c1_pt/wi4ukdci
# dataset.td_prob=0 dataset.role_n_congruence=0 dataset.n_back=5 dataset.concurrent_reg=5 model.model_class=lstm trainer.learning_rate=1e-3 model.from_pretrained=model_checkpoints/bk1zoetn
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.1c1_pt/ku13for7

# dataset.td_prob=1 dataset.role_n_congruence=0 dataset.n_back=5 dataset.concurrent_reg=5 model.model_class=lstm trainer.learning_rate=3e-4 model.from_pretrained=model_checkpoints/t4wviuxs
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.1c1_pt/wu3oaj0b
# dataset.td_prob=1 dataset.role_n_congruence=0 dataset.n_back=5 dataset.concurrent_reg=5 model.model_class=lstm trainer.learning_rate=3e-4 model.from_pretrained=model_checkpoints/bk1zoetn
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.1c1_pt/h3p595s5
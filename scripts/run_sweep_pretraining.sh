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

# TRANSFORMER

# dataset.td_prob=0 dataset.role_n_congruence=0 dataset.concurrent_items=4 dataset.n_back=6 dataset.concurrent_reg=6 dataset.seq_len=200 dataset.n_reg=50 dataset.n_items=50 model.model_class=transformer model.n_layers=2 model.n_heads=4 model.d_model=256 model.d_head=256 model.positional_embedding_type=rotary trainer.learning_rate=0.00022
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.2/56ck9q07
# dataset.td_prob=1 dataset.role_n_congruence=0 dataset.concurrent_items=4 dataset.n_back=6 dataset.concurrent_reg=6 dataset.seq_len=200 dataset.n_reg=50 dataset.n_items=50 model.model_class=transformer model.n_layers=2 model.n_heads=4 model.d_model=256 model.d_head=256 model.positional_embedding_type=rotary trainer.learning_rate=0.00022
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.2/nvgts50y

# dataset.td_prob=0 dataset.role_n_congruence=0 dataset.concurrent_items=4 dataset.n_back=5 dataset.concurrent_reg=5 dataset.seq_len=200 dataset.n_reg=50 dataset.n_items=50 model.model_class=transformer model.n_layers=2 model.n_heads=4 model.d_model=256 model.d_head=256 model.positional_embedding_type=rotary trainer.learning_rate=0.00022
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.2/f27m9gxd
# dataset.td_prob=1 dataset.role_n_congruence=0 dataset.concurrent_items=4 dataset.n_back=5 dataset.concurrent_reg=5 dataset.seq_len=200 dataset.n_reg=50 dataset.n_items=50 model.model_class=transformer model.n_layers=2 model.n_heads=4 model.d_model=256 model.d_head=256 model.positional_embedding_type=rotary trainer.learning_rate=0.00022
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.2/2n6xfvhj

# dataset.td_prob=0 dataset.role_n_congruence=0 dataset.concurrent_items=4 dataset.n_back=4 dataset.concurrent_reg=4 dataset.seq_len=200 dataset.n_reg=50 dataset.n_items=50 model.model_class=transformer model.n_layers=2 model.n_heads=4 model.d_model=256 model.d_head=256 model.positional_embedding_type=rotary trainer.learning_rate=0.00022
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.2/tizab07u
# dataset.td_prob=1 dataset.role_n_congruence=0 dataset.concurrent_items=4 dataset.n_back=4 dataset.concurrent_reg=4 dataset.seq_len=200 dataset.n_reg=50 dataset.n_items=50 model.model_class=transformer model.n_layers=2 model.n_heads=4 model.d_model=256 model.d_head=256 model.positional_embedding_type=rotary trainer.learning_rate=0.00022
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.2/vt7krace

# dataset.td_prob=0 dataset.role_n_congruence=0 dataset.concurrent_items=4 dataset.n_back=3 dataset.concurrent_reg=3 dataset.seq_len=200 dataset.n_reg=50 dataset.n_items=50 model.model_class=transformer model.n_layers=2 model.n_heads=4 model.d_model=256 model.d_head=256 model.positional_embedding_type=rotary trainer.learning_rate=0.00022
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.2/mysyoeaj
# dataset.td_prob=1 dataset.role_n_congruence=0 dataset.concurrent_items=4 dataset.n_back=3 dataset.concurrent_reg=3 dataset.seq_len=200 dataset.n_reg=50 dataset.n_items=50 model.model_class=transformer model.n_layers=2 model.n_heads=4 model.d_model=256 model.d_head=256 model.positional_embedding_type=rotary trainer.learning_rate=0.00022
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.2/oeuwahre

# RECURRENT

# dataset.td_prob=0 dataset.role_n_congruence=0 dataset.concurrent_items=4 dataset.n_back=3 dataset.concurrent_reg=3 dataset.seq_len=200 dataset.n_reg=50 dataset.n_items=50 model.model_class=rnn model.n_layers=2 model.d_model=512 trainer.learning_rate=5e-4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.2/ga0wk6os
# dataset.td_prob=0 dataset.role_n_congruence=0 dataset.concurrent_items=4 dataset.n_back=3 dataset.concurrent_reg=3 dataset.seq_len=200 dataset.n_reg=50 dataset.n_items=50 model.model_class=lstm model.n_layers=2 model.d_model=512 trainer.learning_rate=5e-4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.2/12ztfgrc
# dataset.td_prob=1 dataset.role_n_congruence=0 dataset.concurrent_items=4 dataset.n_back=3 dataset.concurrent_reg=3 dataset.seq_len=200 dataset.n_reg=50 dataset.n_items=50 model.model_class=rnn model.n_layers=2 model.d_model=512 trainer.learning_rate=5e-4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.2/9qx84ank
# dataset.td_prob=1 dataset.role_n_congruence=0 dataset.concurrent_items=4 dataset.n_back=3 dataset.concurrent_reg=3 dataset.seq_len=200 dataset.n_reg=50 dataset.n_items=50 model.model_class=lstm model.n_layers=2 model.d_model=512 trainer.learning_rate=5e-4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.2/pcxbhh1p

# dataset.td_prob=0 dataset.role_n_congruence=0 dataset.concurrent_items=4 dataset.n_back=4 dataset.concurrent_reg=4 dataset.seq_len=200 dataset.n_reg=50 dataset.n_items=50 model.model_class=rnn model.n_layers=2 model.d_model=512 trainer.learning_rate=5e-4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.2/sj193ls7
# dataset.td_prob=0 dataset.role_n_congruence=0 dataset.concurrent_items=4 dataset.n_back=4 dataset.concurrent_reg=4 dataset.seq_len=200 dataset.n_reg=50 dataset.n_items=50 model.model_class=lstm model.n_layers=2 model.d_model=512 trainer.learning_rate=5e-4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.2/umdvst8n
# dataset.td_prob=1 dataset.role_n_congruence=0 dataset.concurrent_items=4 dataset.n_back=4 dataset.concurrent_reg=4 dataset.seq_len=200 dataset.n_reg=50 dataset.n_items=50 model.model_class=rnn model.n_layers=2 model.d_model=512 trainer.learning_rate=5e-4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.2/tyes07ya
# dataset.td_prob=1 dataset.role_n_congruence=0 dataset.concurrent_items=4 dataset.n_back=4 dataset.concurrent_reg=4 dataset.seq_len=200 dataset.n_reg=50 dataset.n_items=50 model.model_class=lstm model.n_layers=2 model.d_model=512 trainer.learning_rate=5e-4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.2/0xtazktl

# dataset.td_prob=0 dataset.role_n_congruence=0 dataset.concurrent_items=4 dataset.n_back=5 dataset.concurrent_reg=5 dataset.seq_len=200 dataset.n_reg=50 dataset.n_items=50 model.model_class=rnn model.n_layers=2 model.d_model=512 trainer.learning_rate=5e-4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.2/d5rdpjsu
# dataset.td_prob=0 dataset.role_n_congruence=0 dataset.concurrent_items=4 dataset.n_back=5 dataset.concurrent_reg=5 dataset.seq_len=200 dataset.n_reg=50 dataset.n_items=50 model.model_class=lstm model.n_layers=2 model.d_model=512 trainer.learning_rate=5e-4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.2/1ub2tu16
# dataset.td_prob=1 dataset.role_n_congruence=0 dataset.concurrent_items=4 dataset.n_back=5 dataset.concurrent_reg=5 dataset.seq_len=200 dataset.n_reg=50 dataset.n_items=50 model.model_class=rnn model.n_layers=2 model.d_model=512 trainer.learning_rate=5e-4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.2/ke53ychb
# dataset.td_prob=1 dataset.role_n_congruence=0 dataset.concurrent_items=4 dataset.n_back=5 dataset.concurrent_reg=5 dataset.seq_len=200 dataset.n_reg=50 dataset.n_items=50 model.model_class=lstm model.n_layers=2 model.d_model=512 trainer.learning_rate=5e-4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.2/s9zvtvp0

# dataset.td_prob=0 dataset.role_n_congruence=0 dataset.concurrent_items=4 dataset.n_back=6 dataset.concurrent_reg=6 dataset.seq_len=200 dataset.n_reg=50 dataset.n_items=50 model.model_class=rnn model.n_layers=2 model.d_model=512 trainer.learning_rate=5e-4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.2/lhez2luh
# dataset.td_prob=0 dataset.role_n_congruence=0 dataset.concurrent_items=4 dataset.n_back=6 dataset.concurrent_reg=6 dataset.seq_len=200 dataset.n_reg=50 dataset.n_items=50 model.model_class=lstm model.n_layers=2 model.d_model=512 trainer.learning_rate=5e-4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.2/u53tn5j9
# dataset.td_prob=1 dataset.role_n_congruence=0 dataset.concurrent_items=4 dataset.n_back=6 dataset.concurrent_reg=6 dataset.seq_len=200 dataset.n_reg=50 dataset.n_items=50 model.model_class=rnn model.n_layers=2 model.d_model=512 trainer.learning_rate=5e-4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.2/gwks4bhg
# dataset.td_prob=1 dataset.role_n_congruence=0 dataset.concurrent_items=4 dataset.n_back=6 dataset.concurrent_reg=6 dataset.seq_len=200 dataset.n_reg=50 dataset.n_items=50 model.model_class=lstm model.n_layers=2 model.d_model=512 trainer.learning_rate=5e-4
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.2/uob77a1o


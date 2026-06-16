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

# dataset.td_prob=0 dataset.role_n_congruence=0 dataset.concurrent_items=4 dataset.n_back=5 dataset.concurrent_reg=5 dataset.seq_len=200 dataset.n_reg=50 dataset.n_items=50 model.model_class=transformer model.n_layers=2 model.n_heads=4 model.d_model=256 model.d_head=256 model.positional_embedding_type=rotary model.from_pretrained=model_checkpoints/f27m9gxd trainer.learning_rate=0.00022
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.2/t6wg82ul
# dataset.td_prob=0 dataset.role_n_congruence=0 dataset.concurrent_items=4 dataset.n_back=5 dataset.concurrent_reg=5 dataset.seq_len=200 dataset.n_reg=50 dataset.n_items=50 model.model_class=transformer model.n_layers=2 model.n_heads=4 model.d_model=256 model.d_head=256 model.positional_embedding_type=rotary model.from_pretrained=model_checkpoints/2n6xfvhj trainer.learning_rate=0.00022
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.2/euxm3zzc
# dataset.td_prob=1 dataset.role_n_congruence=0 dataset.concurrent_items=4 dataset.n_back=5 dataset.concurrent_reg=5 dataset.seq_len=200 dataset.n_reg=50 dataset.n_items=50 model.model_class=transformer model.n_layers=2 model.n_heads=4 model.d_model=256 model.d_head=256 model.positional_embedding_type=rotary model.from_pretrained=model_checkpoints/f27m9gxd trainer.learning_rate=0.00022
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.2/3m077ri1
# dataset.td_prob=1 dataset.role_n_congruence=0 dataset.concurrent_items=4 dataset.n_back=5 dataset.concurrent_reg=5 dataset.seq_len=200 dataset.n_reg=50 dataset.n_items=50 model.model_class=transformer model.n_layers=2 model.n_heads=4 model.d_model=256 model.d_head=256 model.positional_embedding_type=rotary model.from_pretrained=model_checkpoints/2n6xfvhj trainer.learning_rate=0.00022
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.2/zqlpcxy8

# dataset.td_prob=0 dataset.role_n_congruence=0 dataset.concurrent_items=4 dataset.n_back=5 dataset.concurrent_reg=5 dataset.seq_len=200 dataset.n_reg=50 dataset.n_items=50 model.model_class=transformer model.n_layers=2 model.n_heads=4 model.d_model=256 model.d_head=256 model.positional_embedding_type=rotary model.from_pretrained=model_checkpoints/tizab07u trainer.learning_rate=0.00022
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.2/f6presy9
# dataset.td_prob=0 dataset.role_n_congruence=0 dataset.concurrent_items=4 dataset.n_back=5 dataset.concurrent_reg=5 dataset.seq_len=200 dataset.n_reg=50 dataset.n_items=50 model.model_class=transformer model.n_layers=2 model.n_heads=4 model.d_model=256 model.d_head=256 model.positional_embedding_type=rotary model.from_pretrained=model_checkpoints/vt7krace trainer.learning_rate=0.00022
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.2/w4heutgb
# dataset.td_prob=1 dataset.role_n_congruence=0 dataset.concurrent_items=4 dataset.n_back=5 dataset.concurrent_reg=5 dataset.seq_len=200 dataset.n_reg=50 dataset.n_items=50 model.model_class=transformer model.n_layers=2 model.n_heads=4 model.d_model=256 model.d_head=256 model.positional_embedding_type=rotary model.from_pretrained=model_checkpoints/tizab07u trainer.learning_rate=0.00022
# python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.2/ww6e6kr2
# dataset.td_prob=1 dataset.role_n_congruence=0 dataset.concurrent_items=4 dataset.n_back=5 dataset.concurrent_reg=5 dataset.seq_len=200 dataset.n_reg=50 dataset.n_items=50 model.model_class=transformer model.n_layers=2 model.n_heads=4 model.d_model=256 model.d_head=256 model.positional_embedding_type=rotary model.from_pretrained=model_checkpoints/vt7krace trainer.learning_rate=0.00022
python3 -m workingmem --wandb.run_sweep --wandb.sweep_id aloxatel/wm-comp-limit-7.4.2/5jaqz740

# RECURRENT

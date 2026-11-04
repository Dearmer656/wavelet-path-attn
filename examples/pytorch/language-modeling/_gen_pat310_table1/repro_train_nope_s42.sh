#!/bin/bash
#SBATCH --job-name=T1rep_train_nope
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat310_table1_repro/nope_s42/%j_train.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:p6000:2
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# PAT-310 Table 1: current-code re-run of the s42 nope medium fine-tune, original cfg (per-device bs 4; 2 GPUs x accum 8 = global 64,
# same per-step sample set as the original 4-GPU run), stopped by timeout after ~1000 steps. Original s42 loss: step 500 1.7093, step 1000 1.6533.
set -euxo pipefail
set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets WANDB_DISABLED=true WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
MASTER_PORT=$(( 21000 + SLURM_JOB_ID % 10000 ))
timeout 75m /cl/work5/hongyu-s/conda/envs/latest_transformers/bin/torchrun --nproc_per_node=2 --master_port="${MASTER_PORT}" ./run_clm.py --model_type gpt2 --tokenizer_name gpt2 --config_name openai-community/gpt2-medium --model_name_or_path /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/gpt2_medium_owt_nope_80k/checkpoint-80000 --dataset_name mix --block_size 512 --do_train --num_train_epochs 10 --logging_steps 500 --save_steps 100000 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --gradient_accumulation_steps 8 --learning_rate 1e-4 --weight_decay 0.0 --warmup_ratio 0.05 --attn_implementation eager --pe_method no_pe --wavelet_router False --router_band_num 8 --scale_range 0 16 --wavelet_mode logit_bias_ctxscale_shift_v0 --wavelet_baseline_use False --use_beta_modulation False --use_soft_wavelet_fox False --single_A_B True --num_harmonics 1 --share_freq_across_heads True --preprocessing_num_workers 8 --ddp_timeout 21600 --seed 42 --overwrite_output_dir --output_dir /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat310_table1_repro/nope_s42 --logging_dir /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat310_table1_repro/nope_s42/tensorboard || echo "stopped (timeout expected)"
grep -o "{'loss': [0-9.]*[^}]*'epoch': [0-9.]*}" /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat310_table1_repro/nope_s42/*_train.txt | head -3; echo "original s42: step500 1.7093 step1000 1.6533"

#!/bin/bash
#SBATCH --job-name=PAT310_MedMixWPE_s45
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/log_file/train/%j_gpt2_medium_owt_mix_wpe_10ep_s45.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:p6000:2
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Wavelet PE GPT-2 medium fine-tune on mix dataset, matching mix_medium_owt_dd_10ep training args.
# Backbone: runs/gpt2_medium_owt_wavelet_pe_elm82_2p6k/checkpoint-80000
# 2×p6000 (24GB), grad_accum=8 → global_bs=2×4×8=64 (same as mix_medium_owt_dd_10ep)

set -euxo pipefail
echo 'Workdir: /project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling'
cd /project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling

set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true
export WANDB_MODE=disabled

MASTER_PORT=$(( 21400 + SLURM_JOB_ID % 1000 ))

echo '================= BEGIN RUN ================='

/cl/work5/hongyu-s/conda/envs/latest_transformers/bin/torchrun \
  --nproc_per_node=2 \
  --master_port="${MASTER_PORT}" \
  ./run_clm.py \
  --model_type gpt2 \
  --tokenizer_name gpt2 \
  --config_name openai-community/gpt2-medium \
  --model_name_or_path runs/gpt2_medium_owt_wavelet_pe_elm82_2p6k/checkpoint-80000 \
  --dataset_name mix \
  --block_size 512 \
  --do_train \
  --num_train_epochs 10 \
  --logging_steps 500 \
  --save_steps 5000 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --weight_decay 0.0 \
  --warmup_ratio 0.05 \
  --attn_implementation eager \
  --pe_method wavelet \
  --relative_type 4 \
  --wavelet_router False \
  --wavelet_mode logit_bias_ctxscale_shift_v0 \
  --wavelet_baseline_use False \
  --use_beta_modulation False \
  --use_soft_wavelet_fox False \
  --analyzer False \
  --single_A_B True \
  --num_harmonics 1 \
  --share_freq_across_heads True \
  --scale_range 0 16 \
  --router_band_num 8 \
  --preprocessing_num_workers 8 \
  --ddp_timeout 21600 \
  --seed 45 \
  --overwrite_output_dir \
  --output_dir runs/mix_medium_owt_wpe_10ep_s45 \
  --logging_dir ./mix_medium_owt_wpe_10ep_s45_log

echo "=== WPE medium OWT mix fine-tune 10ep done ==="

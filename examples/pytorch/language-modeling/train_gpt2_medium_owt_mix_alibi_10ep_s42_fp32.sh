#!/bin/bash
#SBATCH --job-name=MedMixAlibi_fp32
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/log_file/train/%j_gpt2_medium_owt_mix_alibi_10ep_s42_fp32.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:4
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# PAT-164: GPT-2 medium plain-ALiBi fine-tune on mix dataset, fp32 (no --bf16).
# 2026-09-06: cfg aligned with QWAB's own medium mix finetune
# (train_gpt2_medium_owt_mix_WR_10ep.sh) for comparability: block_size=512, 10 epochs,
# lr=1e-4/wd=0.0/warmup_ratio=0.05, per_device_train/eval_batch_size=8 +
# gradient_accumulation_steps=2 (global_bs=8*4*2=64, matching QWAB's own 8/accum2 exactly
# rather than the Rotary-finetune template's 2/accum8). attn_implementation=flash_attention_2
# (QWAB's own finetune uses its native path_attn implementation, not eager -- ALiBi's
# equivalent "native" choice is flash_attention_2, which this project already verified is
# numerically equivalent to eager for ALiBi at eval time, per the ckpt60000 ppl quick-check;
# also matches this checkpoint's own pretrain attn_implementation).
# Backbone: runs/gpt2_medium_owt_alibi_flash_80k_a6000x4/checkpoint-80000 (job 577056).
# Submit with --dependency=afterok:577056 so this only starts once that pretrain finishes.

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
cd "${WORKDIR}"
PRETRAIN_CKPT="${WORKDIR}/runs/gpt2_medium_owt_alibi_flash_80k_a6000x4/checkpoint-80000"
OUT="${WORKDIR}/runs/mix_medium_owt_alibi_10ep_s42_fp32"
mkdir -p "${OUT}"

MASTER_PORT=$(( 23650 + SLURM_JOB_ID % 1000 ))

echo "=== Plain ALiBi medium mix finetune (fp32): 4xa6000 ==="

/cl/work5/hongyu-s/conda/envs/latest_transformers/bin/torchrun \
  --nproc_per_node=4 \
  --master_port="${MASTER_PORT}" \
  ./run_clm.py \
  --model_type gpt2 \
  --tokenizer_name gpt2 \
  --model_name_or_path "${PRETRAIN_CKPT}" \
  --dataset_name mix \
  --block_size 512 \
  --do_train \
  --num_train_epochs 10 \
  --logging_steps 500 \
  --save_steps 5000 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-4 \
  --weight_decay 0.0 \
  --warmup_ratio 0.05 \
  --attn_implementation flash_attention_2 \
  --pe_method alibi \
  --wavelet_router False \
  --router_band_num 8 \
  --scale_range 0 16 \
  --wavelet_mode logit_bias_ctxscale_shift_v0 \
  --wavelet_baseline_use False \
  --use_beta_modulation False \
  --use_soft_wavelet_fox False \
  --single_A_B True \
  --num_harmonics 1 \
  --share_freq_across_heads True \
  --preprocessing_num_workers 8 \
  --ddp_timeout 21600 \
  --seed 42 \
  --overwrite_output_dir \
  --output_dir "${OUT}" \
  --logging_dir "${OUT}/tensorboard"

echo "=== Plain ALiBi medium mix finetune (fp32) done ==="

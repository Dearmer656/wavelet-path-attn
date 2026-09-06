#!/bin/bash
#SBATCH --job-name=MedRotYarn1024
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/log_file/train/%j_gpt2_medium_owt_mix_rotary_yarn_1024_s42_fp32.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:4
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# PAT-164: Rotary+YaRN GPT-2 medium finetune at the TARGET (extended) length 1024, matching
# the YaRN paper's own protocol (Peng et al. 2023, arXiv:2309.00071): 400 finetune steps,
# global batch size 64 -- confirmed via a live check of the paper, not assumed. 1024 chosen
# over the initially-tried 2048 because factor=2 (1024/512) more closely matches the paper's
# own smallest/most common tested extension ratio (e.g. LLaMA2 4096->8192 is also 2x),
# whereas 2048 (factor=4) was originally picked just to match this project's own Table 4
# length grid, not the paper's typical relative scale.
# Starts from the SAME from-scratch Rotary PRETRAIN checkpoint used by the plain 512
# mix-finetune (runs/gpt2_medium_owt_rotary_80k/checkpoint-80000), NOT the already-512-
# finetuned mix checkpoint -- YaRN's method finetunes directly at the target length rather
# than adapting an already-short-context-finetuned model, per the paper's own design
# (the model needs real long-sequence gradient signal, not just a frequency-formula swap
# at eval time -- see the zero-shot YaRN eval, which showed no improvement over plain RoPE
# at L2048: F1=0.0822 vs 0.0805).
# yarn_factor=2 (1024/512, the true pretrain/finetune block_size). fp32 (matching the
# plain-512 finetune's precision, not the bf16 pretrain). Verified XSum ("mix" dataset's
# 11.1% component) has plenty of usable data at this length: 90.0% of its 204,045 train
# examples fit within 1024 tokens (183,698 usable, 20,319 discarded for being longer) --
# not a data-availability concern, though most XSum examples (median length 423) are still
# mostly padding at this block_size; the real long-context training signal comes from the
# HotpotQA portion (88.9% of "mix"), which is deliberately packed to fill block_size via
# build_context_budgeted.
# 2026-09-06: lr/adam_beta2 corrected to match the YaRN paper's own s=16 finetune recipe
# (verified via live search, not assumed): lr=2e-5 (was 1e-4, this project's baseline
# convention, 5x too high), adam_beta2=0.95 (was HF default 0.999). weight_decay=0.0 and
# warmup_ratio=0.05 (=20/400 steps) already matched the paper's values.

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
PRETRAIN_CKPT="${WORKDIR}/runs/gpt2_medium_owt_rotary_80k/checkpoint-80000"
OUT="${WORKDIR}/runs/mix_medium_owt_rotary_yarn_1024_s42_fp32"
mkdir -p "${OUT}"

MASTER_PORT=$(( 23800 + SLURM_JOB_ID % 1000 ))

echo "=== Rotary+YaRN medium mix finetune @ L1024 (fp32, 400 steps per YaRN paper): 4xa6000 ==="

/cl/work5/hongyu-s/conda/envs/latest_transformers/bin/torchrun \
  --nproc_per_node=4 \
  --master_port="${MASTER_PORT}" \
  ./run_clm.py \
  --model_type gpt2 \
  --tokenizer_name gpt2 \
  --model_name_or_path "${PRETRAIN_CKPT}" \
  --dataset_name mix \
  --block_size 1024 \
  --do_train \
  --max_steps 400 \
  --logging_steps 20 \
  --save_steps 400 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.05 \
  --adam_beta2 0.95 \
  --attn_implementation eager \
  --pe_method rotary \
  --use_yarn True \
  --yarn_factor 2 \
  --yarn_original_max_position_embeddings 512 \
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

echo "=== Rotary+YaRN medium mix finetune @ L1024 done ==="

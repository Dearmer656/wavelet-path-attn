#!/bin/bash
#SBATCH --job-name=alibi_med_80k
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/log_file/train/%j_alibi_med_80k.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:4
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# PAT-164: Plain ALiBi (no QWAB) GPT-2 medium OWT pretrain from scratch, accelerated via
# flash_attention_2's native alibi_slopes support (fla-linear-attention transformers fork,
# modeling_gpt2.py flash_attention_2_alibi_forward). Same recipe as
# train_gpt2_medium_owt_rotary_80k_6000x4.sh for direct comparability:
#   max_steps=80000, warmup_ratio=0.05, eval_steps=5000, save_steps=10000
#   load_best=True, no early_stopping, global_bs=64 (per_device_bs=4, grad_accum=4)
# flash_attention_2_alibi_forward requires an Ampere-or-newer GPU (compute capability >=8.0).
# No fully-idle 4x "6000"/a6000 pool at submit time (all elm71-73/elm61-67 showed MIXED);
# elm54 had a fully-idle 3090x4 (Ampere, cc 8.6 -- satisfies the >=8.0 check) so routed there
# instead of queuing. Do NOT reroute to elm26 -- its "6000" label is a GPU_info.sh mislabel for
# the real gres "q6000" (Turing, cc 7.5), which flash_attention_2_alibi_forward would reject.
# Verified correct via a real eager-vs-flash GPU numerical check (job 576902, a6000):
# shapes match exactly [B,T,H,D], mean/std nearly identical (0.001004/0.389557 vs
# 0.001006/0.389555), max abs diff 0.0158 (bf16 rounding), allclose(atol=2e-2) passed.

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
OUT="${WORKDIR}/runs/gpt2_medium_owt_alibi_flash_80k"
mkdir -p "${OUT}/train"

MASTER_PORT=$(( 24300 + SLURM_JOB_ID % 1000 ))

echo "=== Plain ALiBi (flash_attention_2) medium OWT pretrain: 4x3090 ==="

/cl/work5/hongyu-s/conda/envs/latest_transformers/bin/torchrun \
  --nproc_per_node=4 \
  --master_port="${MASTER_PORT}" \
  ./run_clm.py \
  --model_type gpt2 \
  --config_name openai-community/gpt2-medium \
  --tokenizer_name gpt2 \
  --dataset_name openwebtext \
  --validation_split_percentage 1 \
  --block_size 512 \
  --do_train \
  --do_eval \
  --max_steps 80000 \
  --eval_strategy steps \
  --eval_steps 5000 \
  --logging_steps 500 \
  --save_steps 10000 \
  --load_best_model_at_end True \
  --metric_for_best_model eval_loss \
  --greater_is_better False \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --warmup_ratio 0.05 \
  --bf16 True \
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
  --logging_dir "${OUT}/train/tensorboard"

echo "=== Plain ALiBi (flash_attention_2) medium OWT pretrain done ==="

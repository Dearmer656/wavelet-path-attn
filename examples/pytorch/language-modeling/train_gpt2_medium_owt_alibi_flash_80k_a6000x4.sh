#!/bin/bash
#SBATCH --job-name=alibi_med_80k_a6k
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/log_file/train/%j_alibi_med_80k_a6k.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:4
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# PAT-164: same ALiBi flash_attention_2 medium OWT pretrain, on 4x a6000 (Ampere, cc 8.6 --
# satisfies flash_attention_2_alibi_forward's cc>=8.0 requirement). elm71-73 ("6000") were
# fully allocated while elm64/elm65 each had 4 free a6000 GPUs, so routed here instead of
# waiting further on 576908 (the 6000-pinned speed-comparison job, left pending separately).
# Not a same-hardware match for the cancelled Rotary-on-6000 baseline (575926, 1.20s/it) --
# this is a separate data point, not the strict flash-vs-eager same-GPU comparison.
# Separate output_dir; does not touch 576903/576908/576967.

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
OUT="${WORKDIR}/runs/gpt2_medium_owt_alibi_flash_80k_a6000x4"
mkdir -p "${OUT}/train"

MASTER_PORT=$(( 24300 + SLURM_JOB_ID % 1000 ))

echo "=== Plain ALiBi (flash_attention_2) medium OWT pretrain: 4xa6000 ==="

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

echo "=== Plain ALiBi (flash_attention_2) medium OWT pretrain done (4xa6000) ==="

#!/bin/bash
#SBATCH --job-name=rot_qwab_80k
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/log_file/train/%j_rot_qwab_med_80k.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:4
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# PAT-164: Rotary + QWAB GPT-2 medium OWT pretrain from scratch
# 对齐 gpt2_medium_owt_pytorch_level_path_attn 实际训练设定:
#   max_steps=80000, warmup_ratio=0.05, eval_steps=5000, save_steps=10000
#   load_best=True, no early_stopping, global_bs=64
# 4×A6000 (48GB): per_device_bs=16, grad_accum=1 -> global_bs=64
# QWABBias 通过 pe_method=rotary + wavelet_router=True 激活

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true
export WANDB_MODE=disabled

WORKDIR=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
cd "${WORKDIR}"
OUT="${WORKDIR}/runs/gpt2_medium_owt_rotary_qwab_80k"
mkdir -p "${OUT}/train"

MASTER_PORT=$(( 24200 + SLURM_JOB_ID % 1000 ))

echo "=== Rotary+QWAB medium OWT pretrain: 4×A100 ==="

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
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --warmup_ratio 0.05 \
  --attn_implementation eager \
  --pe_method rotary \
  --wavelet_router True \
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

echo "=== Rotary+QWAB medium OWT pretrain done ==="

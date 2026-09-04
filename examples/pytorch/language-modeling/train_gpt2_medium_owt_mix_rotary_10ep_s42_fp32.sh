#!/bin/bash
#SBATCH --job-name=MedMixRotary_fp32
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/log_file/train/%j_gpt2_medium_owt_mix_rotary_10ep_s42_fp32.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:4
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# PAT-164: GPT-2 medium plain-Rotary fine-tune on mix dataset, fp32 (no --bf16).
# Direct counterpart of the medium PA-only mix finetune
# (train_gpt2_medium_owt_mix_PA_10ep.sh: block_size=512, 10 epochs, global_bs=64,
# warmup_ratio=0.05, no eval/early-stopping -- mirrored here 1:1 except
# attn_implementation/pe_method swapped to eager+rotary).
# Backbone: runs/gpt2_medium_owt_rotary_80k/checkpoint-80000 (submitted with
# --dependency=afterok:575926 so this only starts once that pretrain finishes).
# Batch size cut vs the PA-only script (8/accum2 -> 2/accum8, same global_bs=64):
# eager attention in pure fp32 (no bf16 autocast halving activation memory) is
# heavier than path_attn's O(T) memory profile; the pretrain stage already needed
# bs=4/accum4 *with* bf16 on the same 4x6000(48GB) pool, so fp32 finetune needs a
# smaller per-device batch for headroom.

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
OUT="${WORKDIR}/runs/mix_medium_owt_rotary_10ep_s42_fp32"
mkdir -p "${OUT}"

MASTER_PORT=$(( 23600 + SLURM_JOB_ID % 1000 ))

echo "=== Plain Rotary medium mix finetune (fp32): 4x6000 ==="

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
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --weight_decay 0.0 \
  --warmup_ratio 0.05 \
  --attn_implementation eager \
  --pe_method rotary \
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

echo "=== Plain Rotary medium mix finetune (fp32) done ==="

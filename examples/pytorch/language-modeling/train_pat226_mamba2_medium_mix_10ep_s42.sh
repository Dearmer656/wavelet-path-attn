#!/bin/bash
#SBATCH --job-name=mamba2_med_mixft
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/log_file/train/%j_mamba2_medium_mix_10ep_s42.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:4
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# PAT-164/PAT-226: Mamba-2 medium (370M) mix-dataset finetune, following the pretrain
# (train_pat226_mamba2_medium_6000x4.sh, job 577955).
# 2026-09-06: cfg deliberately kept UNIFORM with every other baseline's finetune stage
# (train_gpt2_medium_owt_mix_WR_10ep.sh / _rotary_10ep_ / _alibi_10ep_): lr=1e-4,
# weight_decay=0.0, warmup_ratio=0.05, block_size=512, 10 epochs, global_bs=64
# (per_device_train/eval_batch_size=8, gradient_accumulation_steps=2, matching QWAB's own
# 8/accum2 exactly) -- explicitly NOT scaled up to Mamba2's own pretrain lr (3e-4).
# Rationale (per explicit discussion): the pretrain-stage lr bump to 3e-4 was needed
# because from-scratch training at this scale plateaued at the project's uniform 1e-4
# (job 577420); finetuning starts from an already-converged checkpoint, so the usual
# lr-sensitivity concern is far weaker, and keeping the finetune-stage recipe uniform
# across all baselines preserves the "vary only architecture" controlled comparison that
# the pretrain-stage deviation already had to give up.
# Backbone: runs/pat226_mamba2/medium_owt_6000x4/checkpoint-80000 (job 577955).
# Submit with --dependency=afterok:577955 so this only starts once that pretrain finishes.
# Uses mamba2_env (real CUDA causal_conv1d/mamba_ssm kernels), same as the pretrain.

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate mamba2_env; set -u
fi

WORKDIR=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
cd "${WORKDIR}"

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PRETRAIN_CKPT="${WORKDIR}/runs/pat226_mamba2/medium_owt_6000x4/checkpoint-80000"
OUT="${WORKDIR}/runs/mix_medium_owt_mamba2_10ep_s42"
mkdir -p "${OUT}"

MASTER_PORT=$(( 24700 + SLURM_JOB_ID % 1000 ))

echo "=== Mamba2 medium mix finetune: 4x6000 ==="

python -m torch.distributed.run --nproc_per_node=4 --master_port=${MASTER_PORT} ./run_clm.py \
  --model_type mamba2 --tokenizer_name gpt2 \
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
  --preprocessing_num_workers 8 \
  --ddp_timeout 21600 \
  --seed 42 \
  --overwrite_output_dir \
  --output_dir "${OUT}" \
  --logging_dir "${OUT}/tensorboard"

echo "=== Mamba2 medium mix finetune done ==="

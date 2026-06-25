#!/bin/bash
#SBATCH --job-name=pat209_train
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_pat209_train.txt
#SBATCH --partition=gpu_long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:6000:1
#SBATCH --time=100:00:00

# PAT-209 step-2: 3-way controlled finetune from rotary checkpoint-15900.
# POS_MODE in {none (baseline RoPE FT), pose_shared, dim_specific}. Identical recipe to
# PAT-204 baseline finetune; only the position-exposure env differs across the 3 runs.
POS_MODE="${POS_MODE:-none}"
POS_TARGET="${POS_TARGET:-4096}"
POS_BUFFER="${POS_BUFFER:-32}"
STEPS="${STEPS:-3000}"
LR="${LR:-5e-5}"

_slack() {
    python3 /project/nlp-work5/hongyu-s/gate1/scripts/notify_slack.py \
        --exit-code "$1" --job-id "${SLURM_JOB_ID}" --node "${SLURMD_NODENAME}" \
        --issue "PAT-209" --gpu "1x6000" \
        --summary "PAT-209 step-2 finetune pos_mode=${POS_MODE} steps=${STEPS}" 2>/dev/null || true
}
trap '_slack $?' EXIT
set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
    set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi
export PYTHONPATH=/cl/work5/hongyu-s/flash-linear-attention:/cl/work5/hongyu-s/transformers/src${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
# PAT-209 position-exposure (read by GPT2Model._maybe_build_pos_exposure; training-only)
export POS_EXPOSURE_MODE=${POS_MODE}
export POS_EXPOSURE_TARGET=${POS_TARGET}
export POS_EXPOSURE_BUFFER=${POS_BUFFER}
# reuse the PAT-204 launcher purely as a no-op pass-through to run_clm (no NoPE)
export NOPE_PRESET=none

cd /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
mkdir -p hotpot_long/logs
CKPT=runs/rotary_mix_finetune/s42/checkpoint-15900
OUT=runs/pat209_pos_exposure/${OUTNAME:-${POS_MODE}}

python hotpot_long/finetune_nope_launcher.py \
    --model_name_or_path ${CKPT} --tokenizer_name gpt2 \
    --dataset_name mix --pe_method rotary --attn_implementation eager \
    --block_size 512 \
    --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps ${GA:-4} \
    --max_steps ${STEPS} --learning_rate ${LR} --weight_decay 0.01 --warmup_ratio 0.05 \
    --do_train \
    --logging_steps 100 --save_steps ${STEPS} --save_total_limit 1 \
    --output_dir ${OUT} --overwrite_output_dir \
    --share_freq_across_heads True --wavelet_router False \
    --wavelet_mode logit_bias_ctxscale_shift_v0 --scale_range 0 16 \
    --analyzer False --router_band_num 8 --use_beta_modulation False \
    --use_soft_wavelet_fox False --wavelet_baseline_use False \
    --single_A_B True --num_harmonics 1 --seed 42

echo "=== Done: PAT-209 finetune pos_mode=${POS_MODE} -> ${OUT} ==="

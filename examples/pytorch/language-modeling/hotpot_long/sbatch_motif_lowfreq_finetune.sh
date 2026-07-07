#!/bin/bash
#SBATCH --job-name=motif_lf_ft
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_motif_lf_ft.txt
#SBATCH --partition=gpu_long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=100:00:00

# PAT-217 x PAT-208: finetune with LOW-FREQ-dim NoPE (all layers, P>cutoff) + low-freq
# RESIDUAL motif on all heads. Bar to beat = PAT-208 low-freq-NoPE-ft (no motif) 0.741/0.698.
#   MODE=real       -> low-freq NoPE + residual motif   (the new method)
#   MODE=nope_only  -> low-freq NoPE, NO motif          (reproduces PAT-208 0.741; control)
MODE="${MODE:-real}"
LAM="${LAM:-1.0}"
PCUT="${PCUT:-512}"
STEPS="${STEPS:-3000}"
LR="${LR:-5e-5}"
LAYERS="${LAYERS:-0-1-2-3-4-5-6-7-8-9-10-11}"
OUTNAME="${OUTNAME:-motif_lf_${MODE}_P${PCUT}}"

_slack() {
    python3 /project/nlp-work5/hongyu-s/gate1/scripts/notify_slack.py \
        --exit-code "$1" --job-id "${SLURM_JOB_ID}" --node "${SLURMD_NODENAME}" \
        --issue "PAT-217" --gpu "1xa6000" \
        --summary "PAT-217 low-freq motif FT ${OUTNAME} steps=${STEPS}" 2>/dev/null || true
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

BASE=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
OUT_D=${BASE}/hotpot_long/analysis_outputs/distill_bias
export MOTIF_NPZ=${OUT_D}/distilled_bias_lowfreq_L512.npz
export MOTIF_MODE=${MODE}
export MOTIF_LAM=${LAM}
export MOTIF_LAYERS=${LAYERS}
export MOTIF_LAM_MODE=const
export MOTIF_DIM_SELECTIVE=1
export MOTIF_PERIOD_CUTOFF=${PCUT}

cd "${BASE}"
mkdir -p hotpot_long/logs
CKPT=runs/rotary_mix_finetune/s42/checkpoint-15900
OUT=runs/pat217_motif_ft/${OUTNAME}

python hotpot_long/motif_launcher.py \
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

echo "=== Done: PAT-217 low-freq motif FT ${OUTNAME} -> ${OUT} ==="

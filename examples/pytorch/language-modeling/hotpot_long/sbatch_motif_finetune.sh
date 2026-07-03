#!/bin/bash
#SBATCH --job-name=motif_ft
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_motif_ft.txt
#SBATCH --partition=gpu_long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:6000:1
#SBATCH --time=100:00:00

# PAT-217 L3: finetune with motif substitution on broken heads (real / none / control).
MODE="${MODE:-real}"
LAM="${LAM:-1.0}"
TOPK="${TOPK:-16}"
STEPS="${STEPS:-3000}"
LR="${LR:-5e-5}"
OUTNAME="${OUTNAME:-motif_${MODE}_lam${LAM}_k${TOPK}}"

_slack() {
    python3 /project/nlp-work5/hongyu-s/gate1/scripts/notify_slack.py \
        --exit-code "$1" --job-id "${SLURM_JOB_ID}" --node "${SLURMD_NODENAME}" \
        --issue "PAT-217" --gpu "1x6000" \
        --summary "PAT-217 L3 finetune motif ${OUTNAME} steps=${STEPS}" 2>/dev/null || true
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
export MOTIF_NPZ=${OUT_D}/distilled_bias_L512.npz
export MOTIF_RECON_CSV=${OUT_D}/recon_error_L2048.csv
export MOTIF_TOPK=${TOPK}
export MOTIF_LAM=${LAM}
export MOTIF_MODE=${MODE}

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

echo "=== Done: PAT-217 L3 finetune ${OUTNAME} -> ${OUT} ==="

#!/bin/bash
#SBATCH --job-name=pat204_plot_attn
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_pat204_plot_attn.txt
#SBATCH --partition=gpu_short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:3090:1
#SBATCH --time=2:00:00

# PAT-204 viz: 12x12 attention-map grid at L2048 for the best-result group
# (finetuned NoPE on layers 8-11), with the rotary baseline as contrast.
CASE_IDX="${CASE_IDX:-0}"
L="${L:-2048}"

_slack() {
    python3 /project/nlp-work5/hongyu-s/gate1/scripts/notify_slack.py \
        --exit-code "$1" --job-id "${SLURM_JOB_ID}" --node "${SLURMD_NODENAME}" \
        --issue "PAT-204" --gpu "1x3090" \
        --summary "PAT-204 plot 12x12 attn L=${L} case=${CASE_IDX} (NoPE8-11 vs baseline)" 2>/dev/null || true
}
trap '_slack $?' EXIT
set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
    set +u; . /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi
export PYTHONPATH=/cl/work5/hongyu-s/flash-linear-attention:/cl/work5/hongyu-s/transformers/src${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1

BASE=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
cd "${BASE}/hotpot_long"
mkdir -p logs
JSONL=${BASE}/hotpot_long/data/hotpot_long_dev_uniform.jsonl
OUT=${BASE}/hotpot_long/analysis_outputs/pat204_attn_maps
mkdir -p "${OUT}"

FT=${BASE}/runs/pat204_nope_vertical_ft/layers8to11/checkpoint-3000
BASELINE=${BASE}/runs/rotary_mix_finetune/s42/checkpoint-15900

# (1) best-result group: finetuned NoPE on layers 8-11, NoPE applied at inference
python plot_attention_maps.py \
    --checkpoint "${FT}" --jsonl "${JSONL}" \
    --preset vertical --nope_layers "8-9-10-11" \
    --length "${L}" --case_idx "${CASE_IDX}" \
    --tag "FT-NoPE8to11" \
    --out "${OUT}/case${CASE_IDX}_L${L}_ftNoPE8to11.png"

# (2) contrast: rotary baseline, no NoPE
python plot_attention_maps.py \
    --checkpoint "${BASELINE}" --jsonl "${JSONL}" \
    --preset none \
    --length "${L}" --case_idx "${CASE_IDX}" \
    --tag "rotary-baseline" \
    --out "${OUT}/case${CASE_IDX}_L${L}_baseline.png"

echo "=== Done: PAT-204 attn maps -> ${OUT} ==="

#!/bin/bash
#SBATCH --job-name=pat209_period_group
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_pat209_period_group.txt
#SBATCH --partition=gpu_short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:3090:1
#SBATCH --time=2:00:00

# PAT-209 step-1 kill-gate: decompose OOD spikes by RoPE period group (short vs long).
CASE_IDX="${CASE_IDX:-0}"
L="${L:-2048}"
TRAIN="${TRAIN:-512}"
CKPT="${CKPT:-runs/rotary_mix_finetune/s42/checkpoint-15900}"

_slack() {
    python3 /project/nlp-work5/hongyu-s/gate1/scripts/notify_slack.py \
        --exit-code "$1" --job-id "${SLURM_JOB_ID}" --node "${SLURMD_NODENAME}" \
        --issue "PAT-209" --gpu "1x3090" \
        --summary "PAT-209 step-1 period-group decompose case=${CASE_IDX} L=${L} train=${TRAIN}" 2>/dev/null || true
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
cd "${BASE}"
mkdir -p hotpot_long/logs
OUT=${BASE}/hotpot_long/analysis_outputs/pat209_period_group/${MODELNAME:-rotary15900}/case${CASE_IDX}_L${L}
JSONL=${BASE}/hotpot_long/data/hotpot_long_dev_uniform.jsonl

python hotpot_long/rope_period_group_decompose.py \
    --checkpoint "${CKPT}" --jsonl "${JSONL}" \
    --L "${L}" --train "${TRAIN}" --case_idx "${CASE_IDX}" \
    --out "${OUT}"
echo "=== Done: PAT-209 step-1 period-group decompose -> ${OUT} ==="

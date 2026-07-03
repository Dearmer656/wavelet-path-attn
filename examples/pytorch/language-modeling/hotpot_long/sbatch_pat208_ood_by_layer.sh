#!/bin/bash
#SBATCH --job-name=pat208_ood_layer
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_pat208_ood_layer.txt
#SBATCH --partition=gpu_short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:3090:1
#SBATCH --time=2:00:00

# PAT-208 follow-up: per-layer OOD-offset logit "splash explosion" profile.
L="${L:-2048}"
TRAIN="${TRAIN:-512}"
NCASES="${NCASES:-5}"
CKPT="${CKPT:-/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/rotary_mix_finetune/s42/checkpoint-15900}"

_slack() {
    python3 /project/nlp-work5/hongyu-s/gate1/scripts/notify_slack.py \
        --exit-code "$1" --job-id "${SLURM_JOB_ID}" --node "${SLURMD_NODENAME}" \
        --issue "PAT-208" --gpu "1x3090" \
        --summary "PAT-208 OOD-logit-by-layer L=${L} train=${TRAIN} n=${NCASES}" 2>/dev/null || true
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
OUT=${BASE}/hotpot_long/analysis_outputs/pat208_qk_cone/ood_by_layer_L${L}

python rope_ood_logit_by_layer.py \
    --checkpoint "${CKPT}" --jsonl "${JSONL}" \
    --L "${L}" --train "${TRAIN}" --n_cases "${NCASES}" \
    --out "${OUT}"
echo "=== Done: PAT-208 OOD-logit-by-layer -> ${OUT} ==="

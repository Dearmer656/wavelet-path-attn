#!/bin/bash
#SBATCH --job-name=pat222_distill
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_pat222_distill.txt
#SBATCH --partition=gpu_long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=100:00:00

# PAT-222: distill slash+sink motif from TinyLlama-1.1B at training length L=2048.
MODEL="${MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
NCASES="${NCASES:-20}"
PCUT="${PCUT:-2048}"

_slack() {
    python3 /project/nlp-work5/hongyu-s/gate1/scripts/notify_slack.py \
        --exit-code "$1" --job-id "${SLURM_JOB_ID}" --node "${SLURMD_NODENAME}" \
        --issue "PAT-222" --gpu "1xa6000" \
        --summary "PAT-222 TinyLlama motif distill L=2048" 2>/dev/null || true
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
OUT=${BASE}/hotpot_long/analysis_outputs/pat222/tinyllama
JSONL=${BASE}/hotpot_long/data/ruler_eval_2048.jsonl

cd "${BASE}/hotpot_long"
mkdir -p logs "${OUT}"

python llm_motif_distill.py \
    --model "${MODEL}" \
    --jsonl "${JSONL}" \
    --L 2048 \
    --n_cases "${NCASES}" \
    --period_cutoff "${PCUT}" \
    --out "${OUT}"

echo "=== Done: PAT-222 distill → ${OUT}/llm_motif_L2048.npz ==="

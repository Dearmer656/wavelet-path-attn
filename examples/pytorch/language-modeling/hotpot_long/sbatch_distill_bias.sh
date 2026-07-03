#!/bin/bash
#SBATCH --job-name=distill_bias
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_distill_bias.txt
#SBATCH --partition=gpu_short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:3090:1
#SBATCH --time=3:00:00

# Step 1: distill L512 logit motif (slash+sink). Step 1.5: L2048 recon-error head ranking.
NCASES="${NCASES:-20}"
CKPT="${CKPT:-/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/rotary_mix_finetune/s42/checkpoint-15900}"

_slack() {
    python3 /project/nlp-work5/hongyu-s/gate1/scripts/notify_slack.py \
        --exit-code "$1" --job-id "${SLURM_JOB_ID}" --node "${SLURMD_NODENAME}" \
        --issue "PAT-217" --gpu "1x3090" \
        --summary "distill L512 motif + L2048 recon-error head ranking" 2>/dev/null || true
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
OUT=${BASE}/hotpot_long/analysis_outputs/distill_bias

python distill_bias_step1.py --checkpoint "${CKPT}" --jsonl "${JSONL}" \
    --L 512 --n_cases "${NCASES}" --out "${OUT}"
python distill_recon_error.py --checkpoint "${CKPT}" --jsonl "${JSONL}" \
    --motif "${OUT}/distilled_bias_L512.npz" --L 2048 --n_cases "${NCASES}" --out "${OUT}"
echo "=== Done: distill bias step1 + recon-error -> ${OUT} ==="

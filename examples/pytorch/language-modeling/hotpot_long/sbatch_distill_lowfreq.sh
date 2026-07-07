#!/bin/bash
#SBATCH --job-name=distill_lf
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_distill_lowfreq.txt
#SBATCH --partition=gpu_long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=100:00:00

# PAT-217 x PAT-208: distill the LOW-FREQ residual structural motif (s(δ)+v(j)) at L512.
CKPT="${CKPT:-/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/rotary_mix_finetune/s42/checkpoint-15900}"
NCASES="${NCASES:-20}"
PCUT="${PCUT:-512}"

_slack() {
    python3 /project/nlp-work5/hongyu-s/gate1/scripts/notify_slack.py \
        --exit-code "$1" --job-id "${SLURM_JOB_ID}" --node "${SLURMD_NODENAME}" \
        --issue "PAT-217" --gpu "1xa6000" \
        --summary "PAT-217 low-freq residual distill (P>${PCUT})" 2>/dev/null || true
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
cd "${BASE}/hotpot_long"
mkdir -p logs
OUT=${BASE}/hotpot_long/analysis_outputs/distill_bias
JSONL=${BASE}/hotpot_long/data/hotpot_long_dev_uniform.jsonl

python distill_bias_lowfreq.py --checkpoint "${CKPT}" --jsonl "${JSONL}" \
    --L 512 --n_cases "${NCASES}" --period_cutoff "${PCUT}" --out "${OUT}"
echo "=== Done: PAT-217 low-freq residual distill -> ${OUT}/distilled_bias_lowfreq_L512.npz ==="

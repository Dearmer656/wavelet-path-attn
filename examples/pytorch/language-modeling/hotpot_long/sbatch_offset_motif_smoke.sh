#!/bin/bash
#SBATCH --job-name=offset_motif
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_offset_motif.txt
#SBATCH --partition=gpu_short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:3090:1
#SBATCH --time=3:00:00

# Gate-1 smoke: offset-space (δ) flexible motif decomposition, L512 vs L2048.
NCASES="${NCASES:-20}"
R="${R:-5}"
CKPT="${CKPT:-/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/rotary_mix_finetune/s42/checkpoint-15900}"

_slack() {
    python3 /project/nlp-work5/hongyu-s/gate1/scripts/notify_slack.py \
        --exit-code "$1" --job-id "${SLURM_JOB_ID}" --node "${SLURMD_NODENAME}" \
        --issue "PAT-208" --gpu "1x3090" \
        --summary "offset-motif Gate-1 smoke L512+L2048 n=${NCASES} R=${R}" 2>/dev/null || true
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
OUT=${BASE}/hotpot_long/analysis_outputs/offset_motif

for L in 2048 512; do
  python offset_motif_smoke.py --checkpoint "${CKPT}" --jsonl "${JSONL}" \
      --L "${L}" --n_cases "${NCASES}" --R "${R}" --out "${OUT}"
done
echo "=== Done: offset-motif Gate-1 smoke -> ${OUT} ==="

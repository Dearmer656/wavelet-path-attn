#!/bin/bash
#SBATCH --job-name=pat222_phi2_distill
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_pat222_phi2_distill.txt
#SBATCH --partition=gpu_long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=100:00:00

# PAT-222: Distill slash+sink motif from Phi-2 at L=2048 (native training length).
MODEL="${MODEL:-microsoft/phi-2}"
L="${L:-2048}"
N_CASES="${N_CASES:-20}"

_slack() {
    python3 /project/nlp-work5/hongyu-s/gate1/scripts/notify_slack.py \
        --exit-code "$1" --job-id "${SLURM_JOB_ID}" --node "${SLURMD_NODENAME}" \
        --issue "PAT-222" --gpu "1xa6000" \
        --summary "PAT-222 Phi-2 motif distill L=${L}" 2>/dev/null || true
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
OUT=${BASE}/hotpot_long/analysis_outputs/pat222/phi2

cd "${BASE}/hotpot_long"
mkdir -p logs "${OUT}"

python llm_motif_distill.py \
    --model "${MODEL}" \
    --jsonl "${BASE}/hotpot_long/data/ruler_eval_${L}.jsonl" \
    --L "${L}" \
    --n_cases "${N_CASES}" \
    --period_cutoff "${L}" \
    --out "${OUT}"

echo "=== Done: PAT-222 Phi-2 distill → ${OUT}/llm_motif_L${L}.npz ==="

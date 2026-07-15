#!/bin/bash
#SBATCH --job-name=pat222_llama2_distill_alldims
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_pat222_llama2_distill_alldims.txt
#SBATCH --partition=gpu_long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=100:00:00

# PAT-222: Distill full-dim (all RoPE dims) motif from LLaMA-2-7B-Chat.
# period_cutoff=0 → all dims selected (no lf/hf split).
# Output: llm_motif_L4096_alldims.npz in the alldims subdirectory.
MODEL="${MODEL:-NousResearch/Llama-2-7b-chat-hf}"
L="${L:-4096}"
N_CASES="${N_CASES:-10}"

_slack() {
    python3 /project/nlp-work5/hongyu-s/gate1/scripts/notify_slack.py \
        --exit-code "$1" --job-id "${SLURM_JOB_ID}" --node "${SLURMD_NODENAME}" \
        --issue "PAT-222" --gpu "1xa6000" \
        --summary "PAT-222 LLaMA-2-7B distill all-dims L=${L}" 2>/dev/null || true
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
OUT=${BASE}/hotpot_long/analysis_outputs/pat222/llama2/alldims

cd "${BASE}/hotpot_long"
mkdir -p logs "${OUT}"

JSONL="${BASE}/hotpot_long/data/ruler_v2_${L}.jsonl"

python llm_motif_distill.py \
    --model "${MODEL}" \
    --jsonl "${JSONL}" \
    --L "${L}" \
    --n_cases "${N_CASES}" \
    --period_cutoff 0.0 \
    --dtype bfloat16 \
    --out "${OUT}"

echo "=== Done: PAT-222 all-dims distill → ${OUT}/llm_motif_L${L}.npz ==="

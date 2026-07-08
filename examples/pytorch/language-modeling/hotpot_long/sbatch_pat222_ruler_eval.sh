#!/bin/bash
#SBATCH --job-name=pat222_ruler
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_pat222_ruler.txt
#SBATCH --partition=gpu_long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:4
#SBATCH --time=100:00:00

# PAT-222: RULER evaluation on TinyLlama-1.1B (baseline + motif).
# Run baseline first (no MOTIF_NPZ), then motif (set MOTIF_NPZ).
MODEL="${MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
MOTIF_NPZ="${MOTIF_NPZ:-}"
LENGTHS="${LENGTHS:-512 2048 4096}"
NPROC="${NPROC:-4}"
NO_CACHE="${NO_CACHE:-1}"   # use_cache=False by default (PAT-222 spec)
LAM="${LAM:-1.0}"
N_CASES="${N_CASES:-0}"    # 0=all; set to e.g. 50 for a quick smoke test

_slack() {
    python3 /project/nlp-work5/hongyu-s/gate1/scripts/notify_slack.py \
        --exit-code "$1" --job-id "${SLURM_JOB_ID}" --node "${SLURMD_NODENAME}" \
        --issue "PAT-222" --gpu "4xa6000" \
        --summary "PAT-222 RULER eval TinyLlama L=${LENGTHS}" 2>/dev/null || true
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
OUT=${BASE}/hotpot_long/analysis_outputs/pat222/ruler

cd "${BASE}/hotpot_long"
mkdir -p logs "${OUT}"

NO_CACHE_FLAG=""
[ "${NO_CACHE}" = "1" ] && NO_CACHE_FLAG="--no_cache"

for L in ${LENGTHS}; do
    # L=512 uses ruler_train_512.jsonl (in-distribution sanity check)
    if [ "${L}" = "512" ]; then
        JSONL=${BASE}/hotpot_long/data/ruler_train_${L}.jsonl
    else
        JSONL=${BASE}/hotpot_long/data/ruler_eval_${L}.jsonl
    fi
    echo "=== L=${L} ==="
    MP=$(( 21000 + SLURM_JOB_ID % 3000 + L % 100 ))

    MOTIF_ARG=""
    [ -n "${MOTIF_NPZ}" ] && MOTIF_ARG="--motif_npz ${MOTIF_NPZ}"

    N_CASES_FLAG=""
    [ "${N_CASES}" != "0" ] && N_CASES_FLAG="--n_cases ${N_CASES}"

    python -m torch.distributed.run --nproc_per_node=${NPROC} --master_port=${MP} \
        ruler_motif_eval.py \
        --model "${MODEL}" \
        --jsonl "${JSONL}" \
        --L "${L}" \
        --lam "${LAM}" \
        ${MOTIF_ARG} ${NO_CACHE_FLAG} ${N_CASES_FLAG} \
        --out "${OUT}"
    echo "--- done L=${L} ---"
done
echo "=== Done: PAT-222 RULER eval ==="

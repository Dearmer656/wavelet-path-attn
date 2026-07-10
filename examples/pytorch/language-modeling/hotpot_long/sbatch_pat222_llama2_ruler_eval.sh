#!/bin/bash
#SBATCH --job-name=pat222_llama2_ruler
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_pat222_llama2_ruler.txt
#SBATCH --partition=gpu_long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:4
#SBATCH --time=100:00:00

# PAT-222: RULER eval on LLaMA-2-7B-Chat with chat template.
MODEL="${MODEL:-NousResearch/Llama-2-7b-chat-hf}"
MOTIF_NPZ="${MOTIF_NPZ:-}"
LENGTHS="${LENGTHS:-4096 8192}"
NPROC="${NPROC:-4}"
LAM="${LAM:-1.0}"
N_CASES="${N_CASES:-0}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-0}"
JSONL_PREFIX="${JSONL_PREFIX:-ruler_eval}"   # ruler_eval (v1) or ruler_v2
OUT_TAG="${OUT_TAG:-}"                        # optional subdir for outputs
EXTRA_FLAGS="${EXTRA_FLAGS:-}"                # e.g. --motif_no_nope / --motif_no_bias

_slack() {
    python3 /project/nlp-work5/hongyu-s/gate1/scripts/notify_slack.py \
        --exit-code "$1" --job-id "${SLURM_JOB_ID}" --node "${SLURMD_NODENAME}" \
        --issue "PAT-222" --gpu "4xa6000" \
        --summary "PAT-222 RULER eval LLaMA-2-7B L=${LENGTHS}" 2>/dev/null || true
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
OUT=${BASE}/hotpot_long/analysis_outputs/pat222/llama2/ruler${OUT_TAG:+/${OUT_TAG}}

cd "${BASE}/hotpot_long"
mkdir -p logs "${OUT}"

for L in ${LENGTHS}; do
    JSONL=${BASE}/hotpot_long/data/${JSONL_PREFIX}_${L}.jsonl
    echo "=== L=${L} ==="
    MP=$(( 22000 + SLURM_JOB_ID % 3000 + L % 1000 ))

    MOTIF_ARG=""
    [ -n "${MOTIF_NPZ}" ] && MOTIF_ARG="--motif_npz ${MOTIF_NPZ}"

    N_CASES_FLAG=""
    [ "${N_CASES}" != "0" ] && N_CASES_FLAG="--n_cases ${N_CASES}"

    MAX_INPUT_FLAG=""
    [ "${MAX_INPUT_TOKENS}" != "0" ] && MAX_INPUT_FLAG="--max_input_tokens ${MAX_INPUT_TOKENS}"

    # motif eval needs full-sequence attention (no KV cache); baseline is fine with cache
    NO_CACHE_FLAG=""
    [ -n "${MOTIF_NPZ}" ] && NO_CACHE_FLAG="--no_cache"

    python -m torch.distributed.run --nproc_per_node=${NPROC} --master_port=${MP} \
        ruler_motif_eval.py \
        --model "${MODEL}" \
        --jsonl "${JSONL}" \
        --L "${L}" \
        --lam "${LAM}" \
        --dtype bfloat16 \
        --apply_chat_template \
        ${NO_CACHE_FLAG} ${MOTIF_ARG} ${N_CASES_FLAG} ${MAX_INPUT_FLAG} ${EXTRA_FLAGS} \
        --out "${OUT}"
    echo "--- done L=${L} ---"
done
echo "=== Done: PAT-222 RULER eval LLaMA-2-7B L=${LENGTHS} ==="

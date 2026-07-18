#!/bin/bash
#SBATCH --job-name=hpuniform_mamba2
#SBATCH --output=hotpot_long/logs/%j_hpuniform_eval_mamba2.txt
#SBATCH --partition=gpu_long
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a6000:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Mamba-2 variant of run_eval_hotpot_long_uniform_a6000.sh: omits all PaTH/wavelet-
# only flags (attn_implementation=path_attn, path_attn_impl, cfg_path, etc.) which
# HfArgumentParser accepts but are meaningless for model_type=mamba2 checkpoints.
# AutoConfig/AutoModelForCausalLM resolve the class from the checkpoint's own
# config.json (model_type=mamba2, registered by fla.models via run_clm.py's
# `import fla.models`).
#
# Usage: sbatch run_eval_hotpot_long_uniform_mamba2.sh <CHECKPOINT> <MODEL_NAME> <BLOCK_SIZE> [LENGTHS_FILTER]

set -euxo pipefail

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention${PYTHONPATH:+:${PYTHONPATH}}

CHECKPOINT="${1:?CHECKPOINT required}"
MODEL_NAME="${2:?MODEL_NAME required}"
BLOCK_SIZE="${3:?BLOCK_SIZE required}"
LENGTHS_FILTER="${4:-${BLOCK_SIZE}}"

LANG_MODEL_DIR="/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling"
HOTPOT_LONG_DIR="${LANG_MODEL_DIR}/hotpot_long"
JSONL="${HOTPOT_LONG_DIR}/data/hotpot_long_dev_uniform.jsonl"

echo "Node: $(hostname) | Model: ${MODEL_NAME} | block_size: ${BLOCK_SIZE}"

mkdir -p "${HOTPOT_LONG_DIR}/logs"
OUTPUT_DIR="${HOTPOT_LONG_DIR}/results_uniform/${MODEL_NAME}/L${BLOCK_SIZE}"
mkdir -p "${OUTPUT_DIR}"

cd "${LANG_MODEL_DIR}"

EXTRA_ARGS=""
if [ -n "${LENGTHS_FILTER}" ]; then
    EXTRA_ARGS="--hotpot_long_lengths ${LENGTHS_FILTER}"
fi

MASTER_PORT=$(( 12000 + SLURM_JOB_ID % 10000 ))
/cl/work5/hongyu-s/conda/envs/latest_transformers/bin/torchrun --nproc_per_node=4 --master_port=${MASTER_PORT} ./run_clm.py \
    --tokenizer_name gpt2 \
    --model_name_or_path "${CHECKPOINT}" \
    --dataset_name hotpot_qa \
    --dataset_config_name distractor \
    --hotpot_long_jsonl "${JSONL}" \
    ${EXTRA_ARGS} \
    --do_eval \
    --block_size "${BLOCK_SIZE}" \
    --per_device_eval_batch_size 4 \
    --output_dir "${OUTPUT_DIR}" \
    --overwrite_output_dir \
    --logging_dir "${OUTPUT_DIR}/log" \
    --seed 42 \
    --load_best_model_at_end False

echo "=== Done: ${MODEL_NAME} L${BLOCK_SIZE} (uniform, mamba2) ==="
echo "Results: ${OUTPUT_DIR}"

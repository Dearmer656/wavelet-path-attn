#!/bin/bash
#SBATCH --job-name=hpuniform_eval
#SBATCH --output=hotpot_long/logs/%j_hpuniform_eval_a6000.txt
#SBATCH --partition=gpu_long
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a6000:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Usage: sbatch run_eval_hotpot_long_uniform_a6000.sh <CHECKPOINT> <MODEL_NAME> <BLOCK_SIZE> <CFG_PATH> [LENGTHS_FILTER] [PATH_ATTN_IMPL]
# LENGTHS_FILTER: comma-separated, e.g. "512" or "2048,4096" (default: BLOCK_SIZE)
# PATH_ATTN_IMPL: "triton" for PA-only models, "pytorch" for WR models (default: pytorch)
#
# Uses uniform-distribution GT dataset: hotpot_long_dev_uniform.jsonl
# Fallback when elm73 is unavailable — uses any node with 4x RTX A6000.

set -euxo pipefail

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention${PYTHONPATH:+:${PYTHONPATH}}
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1

CHECKPOINT="${1:?CHECKPOINT required}"
MODEL_NAME="${2:?MODEL_NAME required}"
BLOCK_SIZE="${3:?BLOCK_SIZE required}"
CFG_PATH="${4:?CFG_PATH required}"
LENGTHS_FILTER="${5:-${BLOCK_SIZE}}"   # default: filter to current BLOCK_SIZE only
PATH_ATTN_IMPL="${6:-pytorch}"         # "triton" for PA-only, "pytorch" for WR

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
/cl/work5/hongyu-s/conda/envs/latest_transformers/bin/torchrun --nproc_per_node=3 --master_port=${MASTER_PORT} ./run_clm.py \
    --model_type gpt2 \
    --tokenizer_name gpt2 \
    --model_name_or_path "${CHECKPOINT}" \
    --attn_implementation path_attn \
    --cfg_path "${CFG_PATH}" \
    --dataset_name hotpot_qa \
    --dataset_config_name distractor \
    --hotpot_long_jsonl "${JSONL}" \
    ${EXTRA_ARGS} \
    --do_eval \
    --block_size "${BLOCK_SIZE}" \
    --per_device_eval_batch_size 4 \
    --path_attn_impl "${PATH_ATTN_IMPL}" \
    --output_dir "${OUTPUT_DIR}" \
    --overwrite_output_dir \
    --logging_dir "${OUTPUT_DIR}/log" \
    --seed 42 \
    --path_use_qk_norm false \
    --path_use_low_rank_w true \
    --path_use_w_shortconv false \
    --path_conv_size 3 \
    --path_conv_bias false \
    --num_harmonics 1 \
    --single_A_B True \
    --use_beta_modulation False \
    --use_soft_wavelet_fox False \
    --wavelet_baseline_use False \
    --use_forget_gate False \
    --qk_rotation False \
    --ablate_switch False \
    --wavelet_router False \
    --load_best_model_at_end False

echo "=== Done: ${MODEL_NAME} L${BLOCK_SIZE} (uniform) ==="
echo "Results: ${OUTPUT_DIR}"

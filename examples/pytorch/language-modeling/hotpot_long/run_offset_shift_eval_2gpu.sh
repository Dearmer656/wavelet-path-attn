#!/bin/bash
#SBATCH --job-name=offset_shift_eval
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_offset_shift_eval.txt
#SBATCH --partition=gpu_long
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:6000:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Offset-Shift Stress Test Eval — PAT review requirement
#
# Usage: sbatch run_offset_shift_eval.sh <CHECKPOINT> <MODEL_NAME> <SHIFT_K> <CFG_PATH>
#
# Evaluates a checkpoint on the k-token-shifted L4096 variant of hotpot_long_dev_uniform.
# Compares performance at different absolute evidence positions to test layout coupling.
#
# Example (PA baseline):
#   sbatch run_offset_shift_eval.sh runs/PA_baseline_multi_seeds/token_even_mix_PA_s42/checkpoint-15000 PA_only_s42 256 runs/PA_baseline_multi_seeds/token_even_mix_PA_s42/supply_model.cfg
#
# Example (LW-WR):
#   sbatch run_offset_shift_eval.sh runs/head_wise_scale_selection_vs_layer_wise/layer_wise/sigmoid_exp/s42_delta_detach/checkpoint-15000 sig_delta_detach_s42 256 runs/head_wise_scale_selection_vs_layer_wise/layer_wise/sigmoid_exp/s42_delta_detach/supply_model.cfg

set -euxo pipefail
if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u; fi
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CHECKPOINT="${1:?CHECKPOINT required}"
MODEL_NAME="${2:?MODEL_NAME required}"
SHIFT_K="${3:?SHIFT_K required (0/256/512/1024)}"
CFG_PATH="${4:?CFG_PATH required}"
MAX_EVAL_SAMPLES="${5:-}"
EVAL_BATCH_SIZE="${6:-2}"

LANG_MODEL_DIR="/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling"
HOTPOT_LONG_DIR="${LANG_MODEL_DIR}/hotpot_long"
SHIFTED_JSONL="${HOTPOT_LONG_DIR}/data/hotpot_long_dev_uniform_shift${SHIFT_K}.jsonl"

echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Model: ${MODEL_NAME} | Shift k=${SHIFT_K}"

if [ ! -f "${SHIFTED_JSONL}" ]; then
    echo "Shifted dataset not found: ${SHIFTED_JSONL}"
    echo "Run make_offset_shift_dataset.py first."
    exit 1
fi

mkdir -p "${HOTPOT_LONG_DIR}/logs"
OUTPUT_SUBDIR="shift${SHIFT_K}"
if [ -n "${MAX_EVAL_SAMPLES}" ]; then
    OUTPUT_SUBDIR="shift${SHIFT_K}_n${MAX_EVAL_SAMPLES}"
fi
OUTPUT_DIR="${HOTPOT_LONG_DIR}/results_offset_shift/${MODEL_NAME}/${OUTPUT_SUBDIR}"
mkdir -p "${OUTPUT_DIR}"

# Overlay cfg: original checkpoint cfg + hotpot_respect_doc_order=True, so build_context_budgeted
# actually respects this dataset's (shift-engineered) document order instead of unconditionally
# pinning evidence to the front. Written per-run so the original training cfg is never mutated.
OVERLAY_CFG="${OUTPUT_DIR}/supply_model_respect_doc_order.cfg"
cp "${CFG_PATH}" "${OVERLAY_CFG}"
# Guarantee a trailing newline before appending: some training-generated supply_model.cfg
# files (e.g. PA_baseline_multi_seeds/token_even_mix_PA_s42) have no final newline, which
# would otherwise glue the appended line onto the last existing line (e.g.
# '...hotpot_question_position="later"hotpot_respect_doc_order=True' as ONE corrupted
# value) and silently drop hotpot_respect_doc_order entirely -- read_kv_config would then
# never set it, defaulting to False and reproducing the exact bug this override exists to
# fix, with no error raised anywhere.
echo >> "${OVERLAY_CFG}"
echo 'hotpot_respect_doc_order=True' >> "${OVERLAY_CFG}"
# These March-2026 checkpoints predate PAT-227's wavelet_ctxscale_scale_max_exp config
# knob and never saved it (nor wavelet_ctxscale_k) in config.json, but their real
# learned wavelet_ctx_router weight is shape [9,64] (null-gate + 8 scales) -- i.e. they
# were trained under the pre-PAT-227 implicit default, which the current code's default
# (wavelet_ctxscale_k=8, scale_max_exp=14.0 scalar) no longer supplies consistently,
# causing a hard shape-validation crash. Supply the documented "bit-identical" legacy
# 8-scale grid so these checkpoints load exactly as they did before that knob existed.
if ! grep -q '^wavelet_ctxscale_scale_max_exp=' "${OVERLAY_CFG}"; then
    echo 'wavelet_ctxscale_scale_max_exp=[0,2,4,6,8,10,12,14]' >> "${OVERLAY_CFG}"
fi

cd "${LANG_MODEL_DIR}"

MAX_EVAL_ARGS=()
if [ -n "${MAX_EVAL_SAMPLES}" ]; then
    MAX_EVAL_ARGS=(--max_eval_samples "${MAX_EVAL_SAMPLES}")
fi

MASTER_PORT=$(( 12000 + SLURM_JOB_ID % 10000 ))
/cl/work5/hongyu-s/conda/envs/latest_transformers/bin/torchrun --nproc_per_node=2 --master_port=${MASTER_PORT} ./run_clm.py \
    "${MAX_EVAL_ARGS[@]}" \
    --model_type gpt2 \
    --tokenizer_name gpt2 \
    --model_name_or_path "${CHECKPOINT}" \
    --attn_implementation path_attn \
    --cfg_path "${OVERLAY_CFG}" \
    --dataset_name hotpot_qa \
    --dataset_config_name distractor \
    --hotpot_long_jsonl "${SHIFTED_JSONL}" \
    --hotpot_long_lengths 4096 \
    --do_eval \
    --block_size 4096 \
    --per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
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

echo "=== Done: ${MODEL_NAME} shift-k=${SHIFT_K} ==="
echo "Results: ${OUTPUT_DIR}"

#!/bin/bash
#SBATCH --job-name=hp4096_K1s44c0
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_hp4096_K1_me16_s44_center0_ricker.txt
#SBATCH --partition=gpu_long
#SBATCH --nodelist=elm54
#SBATCH --gres=gpu:3090:4
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# HotpotQA-Long L4096 (uniform) eval for job 545336 (PAT234_K1_me16_s44_c0_ricker, elm54 3090x4).
# No dependency: pinned to the exact same node+gres as the training job, so this
# just queues (PENDING/Resources) until that job finishes and releases the 4x 3090
# GPUs on elm54, then auto-starts. Checkpoint-15000 is this recipe's final save
# (matches the already-finished K1_me16_noC1_s42 sibling run).

set -euxo pipefail

LANG_MODEL_DIR="/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling"
HOTPOT_LONG_DIR="${LANG_MODEL_DIR}/hotpot_long"
JSONL="${HOTPOT_LONG_DIR}/data/hotpot_long_dev_uniform.jsonl"
RUN_OUT="${LANG_MODEL_DIR}/runs/pat234_scale_card/K1_me16_noC1_s44_center0_ricker"
CHECKPOINT="${RUN_OUT}/checkpoint-15000"
CFG_PATH="${RUN_OUT}/supply_model.cfg"
MODEL_NAME="K1_me16_s44_center0_ricker_ckpt15000"
BLOCK_SIZE=4096

echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) | Model: ${MODEL_NAME}"

mkdir -p "${HOTPOT_LONG_DIR}/logs"
OUTPUT_DIR="${HOTPOT_LONG_DIR}/results_uniform/${MODEL_NAME}/L${BLOCK_SIZE}"
mkdir -p "${OUTPUT_DIR}"

cd "${LANG_MODEL_DIR}"

MASTER_PORT=$(( 12000 + SLURM_JOB_ID % 10000 ))
/cl/work5/hongyu-s/conda/envs/latest_transformers/bin/torchrun --nproc_per_node=4 --master_port=${MASTER_PORT} ./run_clm.py \
    --model_type gpt2 \
    --tokenizer_name gpt2 \
    --model_name_or_path "${CHECKPOINT}" \
    --attn_implementation path_attn \
    --cfg_path "${CFG_PATH}" \
    --dataset_name hotpot_qa \
    --dataset_config_name distractor \
    --hotpot_long_jsonl "${JSONL}" \
    --hotpot_long_lengths ${BLOCK_SIZE} \
    --do_eval \
    --block_size ${BLOCK_SIZE} \
    --per_device_eval_batch_size 4 \
    --path_attn_impl pytorch \
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

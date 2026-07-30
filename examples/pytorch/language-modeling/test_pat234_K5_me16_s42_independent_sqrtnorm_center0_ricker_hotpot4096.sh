#!/bin/bash
#SBATCH --job-name=hp4096_K5s42
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_hp4096_K5_me16_s42_independent_sqrtnorm_center0_ricker.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a100:2
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# HotpotQA-Long L4096 (uniform) eval for job 545496 (PAT234_K5_extscale_ind_sqrt,
# K5_me16_256_1024_4096_8192_noC1_s42_independent_sqrtnorm_center0_ricker),
# already COMPLETED with checkpoint-15000 + a final root save. Unlike the K4
# s44 recipe, this training script has no auto-chained eval, so submitting
# separately. No node pinned (a100, per project convention); elm43 is fully
# idle right now so this should start immediately. Flags/batch size mirror
# hotpot_long/run_eval_hotpot_long_uniform_a6000x2_cpus4.sh (the template the
# K4 auto-chain uses), which is this K-family's own eval convention.

set -euxo pipefail

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true
export WANDB_MODE=disabled

LANG_MODEL_DIR="/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling"
HOTPOT_LONG_DIR="${LANG_MODEL_DIR}/hotpot_long"
JSONL="${HOTPOT_LONG_DIR}/data/hotpot_long_dev_uniform.jsonl"
RUN_OUT="${LANG_MODEL_DIR}/runs/pat234_scale_card/K5_me16_256_1024_4096_8192_noC1_s42_independent_sqrtnorm_center0_ricker"
CHECKPOINT="${RUN_OUT}/checkpoint-15000"
CFG_PATH="${RUN_OUT}/supply_model.cfg"
MODEL_NAME="K5_me16_256_1024_4096_8192_independent_sqrtnorm_s42_ckpt15000"
BLOCK_SIZE=4096

echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) | Model: ${MODEL_NAME}"

mkdir -p "${HOTPOT_LONG_DIR}/logs"
OUTPUT_DIR="${HOTPOT_LONG_DIR}/results_uniform/${MODEL_NAME}/L${BLOCK_SIZE}"
mkdir -p "${OUTPUT_DIR}"

cd "${LANG_MODEL_DIR}"

MASTER_PORT=$(( 12000 + SLURM_JOB_ID % 10000 ))
/cl/work5/hongyu-s/conda/envs/latest_transformers/bin/torchrun --nproc_per_node=2 --master_port=${MASTER_PORT} ./run_clm.py \
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
    --per_device_eval_batch_size 2 \
    --path_attn_impl pytorch \
    --report_to none \
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

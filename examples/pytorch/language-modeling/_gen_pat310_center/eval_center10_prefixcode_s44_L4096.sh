#!/bin/bash
#SBATCH --job-name=pat310_c10_s44_L4096
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_pat310_center10_prefixcode_s44_L4096.txt
#SBATCH --partition=lang_gpu_long
#SBATCH --account=lang
#SBATCH --gres=gpu:a100:2
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# PAT-310 W1: eval K1_me16_noC1_s44_wavelet_center10 (trained 07-29/30, pre-fix code) with the SAME pre-fix code
# it was trained/evaluated with: fla 746ce2749c + transformers f6267bfc19 worktrees (before 968a971f81 router-RMS removal,
# d339c6292a p99-clamp removal, e15c9db6a1 g_layer removal). s43 is a reproduction check vs the 07-29 result F1=0.6832.
# Flags identical to test_pat234_K1_me16_s43_wavelet_center10_hotpot4096.sh (bs4 x 2 GPU, seed 42).
set -euxo pipefail
WT=/project/nlp-work5/hongyu-s/prefix_worktrees
export PYTHONPATH=${WT}/tf_f6267bf/src:${WT}/fla_746ce27:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true WANDB_MODE=disabled
LANG_MODEL_DIR="/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling"
HOTPOT_LONG_DIR="${LANG_MODEL_DIR}/hotpot_long"
JSONL="${HOTPOT_LONG_DIR}/data/hotpot_long_dev_uniform.jsonl"
RUN_OUT="${LANG_MODEL_DIR}/runs/pat234_scale_card/K1_me16_noC1_s44_wavelet_center10"
CHECKPOINT="${RUN_OUT}/checkpoint-15000"
CFG_PATH="${RUN_OUT}/supply_model.cfg"
MODEL_NAME="pat310_K1_me16_s44_wavelet_center10_prefixcode_ckpt15000"
BLOCK_SIZE=4096
[ -d "${CHECKPOINT}" ] || { echo "Missing ${CHECKPOINT}" >&2; exit 1; }
OUTPUT_DIR="${HOTPOT_LONG_DIR}/results_uniform/${MODEL_NAME}/L${BLOCK_SIZE}"
mkdir -p "${OUTPUT_DIR}"
cd ${WT}/tf_f6267bf/examples/pytorch/language-modeling
/cl/work5/hongyu-s/conda/envs/latest_transformers/bin/python -c "import fla, transformers; print(fla.__file__, transformers.__file__)"
MASTER_PORT=$(( 12000 + SLURM_JOB_ID % 10000 ))
/cl/work5/hongyu-s/conda/envs/latest_transformers/bin/torchrun --nproc_per_node=2 --master_port=${MASTER_PORT} ./run_clm.py --model_type gpt2 --tokenizer_name gpt2 --model_name_or_path "${CHECKPOINT}" --attn_implementation path_attn --cfg_path "${CFG_PATH}" --dataset_name hotpot_qa --dataset_config_name distractor --hotpot_long_jsonl "${JSONL}" --hotpot_long_lengths ${BLOCK_SIZE} --do_eval --block_size ${BLOCK_SIZE} --per_device_eval_batch_size 4 --path_attn_impl pytorch --output_dir "${OUTPUT_DIR}" --overwrite_output_dir --logging_dir "${OUTPUT_DIR}/log" --seed 42 --path_use_qk_norm false --path_use_low_rank_w true --path_use_w_shortconv false --path_conv_size 3 --path_conv_bias false --num_harmonics 1 --single_A_B True --use_beta_modulation False --use_soft_wavelet_fox False --wavelet_baseline_use False --use_forget_gate False --qk_rotation False --ablate_switch False --wavelet_router False --load_best_model_at_end False
echo "=== Done: ${MODEL_NAME} ==="

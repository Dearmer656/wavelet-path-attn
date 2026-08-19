#!/bin/bash
#SBATCH --job-name=hp4096_K1s43_dcnosqrt
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_hp4096_K1_me16_s43_dual_center_nosqrt.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:2
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
set -euxo pipefail
if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u; fi
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true WANDB_MODE=disabled
LANG_MODEL_DIR="/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling"
HOTPOT_LONG_DIR="${LANG_MODEL_DIR}/hotpot_long"
JSONL="${HOTPOT_LONG_DIR}/data/hotpot_long_dev_uniform.jsonl"
RUN_OUT="${LANG_MODEL_DIR}/runs/pat234_scale_card/K1_me16_noC1_s43_dual_center_seprms_nosqrt"
CHECKPOINT="${RUN_OUT}/checkpoint-15000"
CFG_PATH="${RUN_OUT}/supply_model.cfg"
MODEL_NAME="K1_me16_s43_dual_center_seprms_nosqrt_ckpt15000"
BLOCK_SIZE=4096
[ -d "${CHECKPOINT}" ] || { echo "Missing ${CHECKPOINT}" >&2; exit 1; }
OUTPUT_DIR="${HOTPOT_LONG_DIR}/results_uniform/${MODEL_NAME}/L${BLOCK_SIZE}"
mkdir -p "${OUTPUT_DIR}"
cd "${LANG_MODEL_DIR}"
MASTER_PORT=$(( 12000 + SLURM_JOB_ID % 10000 ))
torchrun --nproc_per_node=2 --master_port=${MASTER_PORT} ./run_clm.py --model_type gpt2 --tokenizer_name gpt2 --model_name_or_path "${CHECKPOINT}" --attn_implementation path_attn --cfg_path "${CFG_PATH}" --dataset_name hotpot_qa --dataset_config_name distractor --hotpot_long_jsonl "${JSONL}" --hotpot_long_lengths ${BLOCK_SIZE} --do_eval --block_size ${BLOCK_SIZE} --per_device_eval_batch_size 2 --path_attn_impl pytorch --report_to none --output_dir "${OUTPUT_DIR}" --overwrite_output_dir --logging_dir "${OUTPUT_DIR}/log" --seed 43 --path_use_qk_norm false --path_use_low_rank_w true --path_use_w_shortconv false --path_conv_size 3 --path_conv_bias false --num_harmonics 1 --single_A_B True --use_beta_modulation False --use_soft_wavelet_fox False --wavelet_baseline_use False --use_forget_gate False --qk_rotation False --ablate_switch False --wavelet_router False --load_best_model_at_end False
echo "=== Done: ${MODEL_NAME} L${BLOCK_SIZE} (uniform) ==="

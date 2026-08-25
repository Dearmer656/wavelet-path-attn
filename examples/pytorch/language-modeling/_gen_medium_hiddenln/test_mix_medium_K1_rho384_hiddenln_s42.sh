#!/bin/bash
#SBATCH --job-name=hp2048_medK1_rho384_hiddenln_s42
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_medK1_rho384_hiddenln_s42_ckpt15000_hotpot2048.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:4
#SBATCH --exclude=elm66
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
set -euxo pipefail
if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u; fi
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
JSONL="/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/hotpot_long_dev_uniform.jsonl"
CHECKPOINT="/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/mix_medium_K1_rho384_hiddenln_s42/checkpoint-15000"
CFG_PATH="/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/mix_medium_K1_rho384_hiddenln_s42/supply_model.cfg"
BIAS_TYPE=$(grep -oP '(?<=bias_type=")[^"]+' "${CFG_PATH}")
BLOCK_SIZE=2048
[ -d "${CHECKPOINT}" ] || { echo "Missing ${CHECKPOINT}" >&2; exit 1; }
OUTPUT_DIR="/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/results_uniform/mix_medium_K1_rho384_hiddenln_s42_ckpt15000/L${BLOCK_SIZE}"
mkdir -p "${OUTPUT_DIR}"; cd "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling"
MASTER_PORT=$((12000 + SLURM_JOB_ID % 10000))
torchrun --nproc_per_node=4 --master_port=${MASTER_PORT} ./run_clm.py --model_type gpt2 --tokenizer_name gpt2 --model_name_or_path "${CHECKPOINT}" --attn_implementation path_attn --bias_type ${BIAS_TYPE} --cfg_path "${CFG_PATH}" --dataset_name hotpot_qa --dataset_config_name distractor --hotpot_long_jsonl "${JSONL}" --hotpot_long_lengths ${BLOCK_SIZE} --do_eval --block_size ${BLOCK_SIZE} --per_device_eval_batch_size 1 --path_attn_impl pytorch --report_to none --output_dir "${OUTPUT_DIR}" --overwrite_output_dir --logging_dir "${OUTPUT_DIR}/log" --seed 42 --path_use_qk_norm false --path_use_low_rank_w true --path_use_w_shortconv false --path_conv_size 3 --path_conv_bias false --num_harmonics 1 --single_A_B True --use_beta_modulation False --use_soft_wavelet_fox False --wavelet_baseline_use False --use_forget_gate False --qk_rotation False --ablate_switch False --wavelet_router False --load_best_model_at_end False
echo "=== Done: mix_medium_K1_rho384_hiddenln_s42 L${BLOCK_SIZE} ==="

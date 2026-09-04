#!/bin/bash
#SBATCH --job-name=ricker128_realbias_L512
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_K1_rho128_ricker_s42_L512_a6000_realbias.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Table 4 (GPT-2 small) QWAB/Ricker rho=128, L512 only, REAL bias (pytorch impl, not triton).
# Split per-length across separate a6000 GPUs per user request (2026-09-04), instead of one
# sequential all-lengths job stuck waiting on DRAINed a100-80 / fully-occupied p6000.
# CAVEAT: L8192+ previously confirmed OOM on a single 47.4GB a6000/6000 even with bf16 (needs
# p6000 96GB / a100-80). This may fail again for the same reason -- worth retrying since per-length
# isolation means a failure here doesn't block the other 5 lengths.
# Same checkpoint as the rest of this row: runs/pat244_dual_temp/K1_L512_me14_rho128_ricker_s42/checkpoint-15000

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
CKPT="${BASE}/runs/pat244_dual_temp/K1_L512_me14_rho128_ricker_s42/checkpoint-15000"
CFG_PATH="${BASE}/runs/pat244_dual_temp/K1_L512_me14_rho128_ricker_s42/supply_model.cfg"
cd "${BASE}"

BSIZE=512
JSONL="${BASE}/hotpot_long/data/hotpot_long_dev_uniform.jsonl"
OUTPUT="${BASE}/hotpot_long/results_uniform/K1_L512_me14_rho128_ricker_s42_ckpt15000_a6000_realbias/L${BSIZE}"
mkdir -p "${OUTPUT}"
echo "=== K1_rho128_ricker s42 L${BSIZE} (a6000, real bias) ==="
MASTER_PORT=$((12000 + SLURM_JOB_ID % 10000 + BSIZE % 100))
torchrun --nproc_per_node=1 --master_port=${MASTER_PORT} ./run_clm.py \
  --model_type gpt2 --tokenizer_name gpt2 \
  --model_name_or_path "${CKPT}" \
  --attn_implementation path_attn --bias_type wavelet \
  --cfg_path "${CFG_PATH}" \
  --dataset_name hotpot_qa --dataset_config_name distractor \
  --hotpot_long_jsonl "${JSONL}" \
  --hotpot_long_lengths ${BSIZE} \
  --do_eval --block_size ${BSIZE} \
  --per_device_eval_batch_size 1 \
  --path_attn_impl pytorch \
  
  --report_to none \
  --output_dir "${OUTPUT}" --overwrite_output_dir \
  --logging_dir "${OUTPUT}/log" \
  --seed 42 \
  --path_use_qk_norm false --path_use_low_rank_w true --path_use_w_shortconv false \
  --path_conv_size 3 --path_conv_bias false --num_harmonics 1 --single_A_B True \
  --use_beta_modulation False --use_soft_wavelet_fox False --wavelet_baseline_use False \
  --use_forget_gate False --qk_rotation False --ablate_switch False --wavelet_router False \
  --load_best_model_at_end False
python3 -c "import json; d=json.load(open('${OUTPUT}/eval_results.json')); print(f'K1_rho128_ricker s42 L${BSIZE} (a6000 real bias): F1={d[\"eval_f1\"]:.4f}')"

echo "=== Done: K1 rho=128 Ricker s42 L512 (a6000, real bias) ==="

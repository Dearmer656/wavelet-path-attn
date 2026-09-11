#!/bin/bash
#SBATCH --job-name=hp_wpe_12288
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_hp_wpe_medium_mix_10ep_12288_elm73.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:7
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# HotpotQA-Long uniform eval for WPE GPT-2 medium, mix 10ep checkpoint-15000
# L=12288 only, elm73 7×RTX6000Ada (48GB each)

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
cd "${WORKDIR}"

CKPT="${WORKDIR}/runs/mix_medium_owt_wpe_10ep/checkpoint-15000"
DATA_DIR="${WORKDIR}/hotpot_long/data"
RESULT_BASE="${WORKDIR}/hotpot_long/results_uniform/wpe_medium_mix_10ep_ckpt15000"
OUT_DIR="${RESULT_BASE}/L12288"
mkdir -p "${OUT_DIR}"

MASTER_PORT=$(( 15000 + SLURM_JOB_ID % 10000 ))

echo "=== WPE medium mix 10ep HotpotQA-Long L12288 | elm73 7×RTX6000Ada | ckpt: ${CKPT} ==="

/cl/work5/hongyu-s/conda/envs/latest_transformers/bin/torchrun \
  --nproc_per_node=7 \
  --master_port="${MASTER_PORT}" \
  ./run_clm.py \
  --model_type gpt2 \
  --tokenizer_name gpt2 \
  --model_name_or_path "${CKPT}" \
  --pe_method wavelet \
  --relative_type 4 \
  --attn_implementation eager \
  --wavelet_router False \
  --router_band_num 8 \
  --scale_range 0 16 \
  --wavelet_mode logit_bias_ctxscale_shift_v0 \
  --wavelet_baseline_use False \
  --use_beta_modulation False \
  --use_soft_wavelet_fox False \
  --single_A_B True \
  --num_harmonics 1 \
  --share_freq_across_heads True \
  --analyzer False \
  --dataset_name hotpot_qa \
  --dataset_config_name distractor \
  --hotpot_long_jsonl "${DATA_DIR}/hotpot_long_dev_uniform_12288only.jsonl" \
  --hotpot_long_lengths 12288 \
  --do_eval \
  --block_size 12288 \
  --per_device_eval_batch_size 1 \
  --output_dir "${OUT_DIR}" \
  --overwrite_output_dir \
  --logging_dir "${OUT_DIR}/log" \
  --load_best_model_at_end False \
  --seed 42

echo "=== Done: $(cat ${OUT_DIR}/eval_results.json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'F1={d[\"eval_f1\"]:.4f} EM={d[\"eval_em\"]:.4f}')" 2>/dev/null || echo 'no results') ==="

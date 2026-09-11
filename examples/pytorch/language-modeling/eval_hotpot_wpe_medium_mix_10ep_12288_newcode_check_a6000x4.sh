#!/bin/bash
#SBATCH --job-name=hp_wpe_12288_newcode
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_hp_wpe_medium_mix_10ep_12288_newcode_check_a6000x4.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:4
#SBATCH --nodelist=elm64
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# OOM check: WRP/WPE L12288 HotpotQA-Long eval under the NEW chunked O(D,L)
# wavelet-relative-tensor implementation (_compute_wavelet_relative_bias),
# on 4x a6000 (48GB each) -- fewer/smaller-parallelism GPUs than the existing
# 7x6000 L12288 script, per-GPU memory footprint is what matters here (each
# rank hosts its own full model + local batch, unaffected by GPU count).
# Old dense code would need [64,12288,12288] fp32 ~= 38.6GB transiently per
# layer per forward call just for the wavelet relative-position tensor.

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
OUT_DIR="${WORKDIR}/hotpot_long/results_uniform/wpe_medium_mix_10ep_ckpt15000_newcode_check/L12288"
mkdir -p "${OUT_DIR}"

MASTER_PORT=$(( 16000 + SLURM_JOB_ID % 10000 ))

echo "=== WPE medium mix 10ep HotpotQA-Long L12288 [NEW CHUNKED CODE, OOM CHECK] | elm64 4x a6000 | ckpt: ${CKPT} ==="
nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv

/cl/work5/hongyu-s/conda/envs/latest_transformers/bin/torchrun \
  --nproc_per_node=4 \
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

echo "=== Done: $(cat ${OUT_DIR}/eval_results.json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'F1={d[\"eval_f1\"]:.4f} EM={d[\"eval_em\"]:.4f} loss={d[\"eval_loss\"]:.4f}')" 2>/dev/null || echo 'no results') ==="

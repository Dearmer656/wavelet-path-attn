#!/bin/bash
#SBATCH --job-name=hp_wpe_16384_newcode
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_hp_wpe_medium_mix_10ep_16384_newcode_check_p6000x1.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:p6000:1
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# OOM check: WRP/WPE L16384 HotpotQA-Long eval under the NEW chunked O(D,L)
# wavelet-relative-tensor implementation, on a SINGLE p6000 (48GB).
# Old dense code needed [64,16384,16384] fp32 ~= 68.7GB transiently per layer
# per forward call -- guaranteed OOM on any single 48GB card. This checks
# whether the new chunked implementation fits.

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
OUT_DIR="${WORKDIR}/hotpot_long/results_uniform/wpe_medium_mix_10ep_ckpt15000_newcode_check/L16384"
mkdir -p "${OUT_DIR}"

MASTER_PORT=$(( 17000 + SLURM_JOB_ID % 10000 ))

echo "=== WPE medium mix 10ep HotpotQA-Long L16384 [NEW CHUNKED CODE, OOM CHECK] | 1x p6000 | ckpt: ${CKPT} ==="
nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv

/cl/work5/hongyu-s/conda/envs/latest_transformers/bin/torchrun \
  --nproc_per_node=1 \
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
  --hotpot_long_jsonl "${DATA_DIR}/hotpot_long_dev_uniform_16384only.jsonl" \
  --hotpot_long_lengths 16384 \
  --do_eval \
  --block_size 16384 \
  --per_device_eval_batch_size 1 \
  --output_dir "${OUT_DIR}" \
  --overwrite_output_dir \
  --logging_dir "${OUT_DIR}/log" \
  --load_best_model_at_end False \
  --seed 42

echo "=== Done: $(cat ${OUT_DIR}/eval_results.json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'F1={d[\"eval_f1\"]:.4f} EM={d[\"eval_em\"]:.4f} loss={d[\"eval_loss\"]:.4f}')" 2>/dev/null || echo 'no results') ==="

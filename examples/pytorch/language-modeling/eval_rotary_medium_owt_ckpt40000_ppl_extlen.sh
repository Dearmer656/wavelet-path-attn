#!/bin/bash
#SBATCH --job-name=rotmed_ppl_extlen
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/gpt2_medium_owt_rotary_80k/train/%j_ppl_extlen_ckpt40000.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Extended-length perplexity check for the plain Rotary GPT-2 medium OWT pretrain
# (PAT-164 parallel work, still training toward 80000 steps) using its current
# intermediate checkpoint (checkpoint-40000). Trained at block_size=512; this tests
# how perplexity degrades at 2x/4x/8x/16x/32x that length (length extrapolation),
# on the same dataset it was pretrained on (OpenWebText validation split).

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

BASE=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
CKPT="${BASE}/runs/gpt2_medium_owt_rotary_80k/checkpoint-40000"
cd "${BASE}"

for BSIZE in 512 1024 2048 4096 8192 16384; do
  OUTPUT="${BASE}/runs/gpt2_medium_owt_rotary_80k/ppl_extlen_ckpt40000/L${BSIZE}"
  mkdir -p "${OUTPUT}"
  echo "=== rotary medium ckpt40000 ppl @ block_size=${BSIZE} ==="
  MASTER_PORT=$(( 13000 + SLURM_JOB_ID % 10000 + BSIZE % 100 ))
  if [ "${BSIZE}" -gt 4096 ]; then
    PRECISION_ARGS="--bf16 True"
  else
    PRECISION_ARGS=""
  fi
  python -m torch.distributed.run --nproc_per_node=1 --master_port=${MASTER_PORT} ./run_clm.py \
    --model_type gpt2 --tokenizer_name gpt2 \
    --model_name_or_path "${CKPT}" \
    --dataset_name openwebtext \
    --validation_split_percentage 1 \
    --pe_method rotary --attn_implementation eager \
    --block_size "${BSIZE}" \
    --do_eval \
    --per_device_eval_batch_size 1 \
    ${PRECISION_ARGS} \
    --wavelet_router False \
    --wavelet_mode logit_bias_ctxscale_shift_v0 \
    --scale_range 0 16 \
    --router_band_num 8 \
    --use_beta_modulation False \
    --use_soft_wavelet_fox False \
    --wavelet_baseline_use False \
    --single_A_B True \
    --num_harmonics 1 \
    --share_freq_across_heads True \
    --output_dir "${OUTPUT}" --overwrite_output_dir \
    --logging_dir "${OUTPUT}/log" \
    --seed 42 --load_best_model_at_end False
  python3 -c "import json; d=json.load(open('${OUTPUT}/eval_results.json')); print(f'rotary medium ckpt40000 L${BSIZE}: eval_loss={d[\"eval_loss\"]:.4f} ppl={d[\"perplexity\"]:.2f}')"
done

echo "=== Done: rotary medium ckpt40000 ppl extended-length sweep ==="

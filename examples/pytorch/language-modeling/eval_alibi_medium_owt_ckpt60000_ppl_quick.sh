#!/bin/bash
#SBATCH --job-name=alibimed_ppl_quick
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/gpt2_medium_owt_alibi_flash_80k_a6000x4/train/%j_ppl_quick_ckpt60000.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:2
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Quick extended-length OWT perplexity spot-check for the ALiBi GPT-2 medium pretrain's
# latest available checkpoint (checkpoint-60000, pretrain still running at ~62k/80k steps).
# Mirrors eval_rotary_medium_owt_ckpt40000_ppl_quick.sh exactly (L1024/L2048, 1000-sample
# cap). Uses plain eager attention (not flash_attention_2) for the eval -- already verified
# this session (job 576902) that eager vs flash_attention_2+alibi match within bf16
# rounding, and eval doesn't need the training-time speed path.

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
CKPT="${BASE}/runs/gpt2_medium_owt_alibi_flash_80k_a6000x4/checkpoint-60000"
cd "${BASE}"

for BSIZE in 1024 2048; do
  OUTPUT="${BASE}/runs/gpt2_medium_owt_alibi_flash_80k_a6000x4/ppl_quick_ckpt60000/L${BSIZE}"
  mkdir -p "${OUTPUT}"
  echo "=== alibi medium ckpt60000 ppl @ block_size=${BSIZE} (1000 samples) ==="
  MASTER_PORT=$(( 13000 + SLURM_JOB_ID % 10000 + BSIZE % 100 ))
  python -m torch.distributed.run --nproc_per_node=2 --master_port=${MASTER_PORT} ./run_clm.py \
    --model_type gpt2 --tokenizer_name gpt2 \
    --model_name_or_path "${CKPT}" \
    --dataset_name openwebtext \
    --validation_split_percentage 1 \
    --max_eval_samples 1000 \
    --preprocessing_num_workers 8 \
    --pe_method alibi --attn_implementation eager \
    --block_size "${BSIZE}" \
    --do_eval \
    --per_device_eval_batch_size 1 \
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
  python3 -c "import json; d=json.load(open('${OUTPUT}/eval_results.json')); print(f'alibi medium ckpt60000 L${BSIZE} (1000 samples): eval_loss={d[\"eval_loss\"]:.4f} ppl={d[\"perplexity\"]:.2f}')"
done

echo "=== Done: alibi medium ckpt60000 quick ppl (L1024, L2048, 1000 samples each) ==="

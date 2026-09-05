#!/bin/bash
#SBATCH --job-name=rotmed_ppl_L4096
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/gpt2_medium_owt_rotary_80k/train/%j_ppl_L4096_ckpt80000.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:2
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Extended-length perplexity spot-check for the plain Rotary GPT-2 medium OWT pretrain's
# FINAL checkpoint-80000 at L4096 (8x train context, block_size=512) -- continues the
# earlier L1024/L2048 quick check (eval_rotary_medium_owt_ckpt40000_ppl_quick.sh, run on
# the intermediate checkpoint-40000) now that the pretrain has finished. Capped at 1000
# eval examples. 2x a6000 (nproc_per_node=2) to split the 1000 samples for wall-clock
# speed, not because L4096 needs more than 1 GPU's memory (per_device_eval_batch_size=1
# throughout, matching the L1024/L2048 precedent).

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
CKPT="${BASE}/runs/gpt2_medium_owt_rotary_80k/checkpoint-80000"
cd "${BASE}"

OUTPUT="${BASE}/runs/gpt2_medium_owt_rotary_80k/ppl_quick_ckpt80000/L4096"
mkdir -p "${OUTPUT}"
MASTER_PORT=$(( 13000 + SLURM_JOB_ID % 10000 ))

echo "=== rotary medium ckpt80000 ppl @ block_size=4096 (1000 samples, 2xa6000) ==="

python -m torch.distributed.run --nproc_per_node=2 --master_port=${MASTER_PORT} ./run_clm.py \
  --model_type gpt2 --tokenizer_name gpt2 \
  --model_name_or_path "${CKPT}" \
  --dataset_name openwebtext \
  --validation_split_percentage 1 \
  --max_eval_samples 1000 \
  --preprocessing_num_workers 8 \
  --pe_method rotary --attn_implementation eager \
  --block_size 4096 \
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

python3 -c "import json; d=json.load(open('${OUTPUT}/eval_results.json')); print(f'rotary medium ckpt80000 L4096 (1000 samples): eval_loss={d[\"eval_loss\"]:.4f} ppl={d[\"perplexity\"]:.2f}')"

echo "=== Done: rotary medium ckpt80000 L4096 quick ppl ==="

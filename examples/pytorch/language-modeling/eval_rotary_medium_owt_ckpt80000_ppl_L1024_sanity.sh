#!/bin/bash
#SBATCH --job-name=rotmed_ppl_L1024_san
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/gpt2_medium_owt_rotary_80k/train/%j_ppl_L1024_ckpt80000_sanity.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Cheap sanity check on checkpoint-80000's eval setup before trusting the concurrently
# running L4096 job (577438, same checkpoint/flags apart from block_size): L1024, 500
# samples, 1 GPU. Reuses the exact same flag set as
# eval_rotary_medium_owt_ckpt80000_ppl_L4096.sh / the earlier ckpt40000 L1024 check
# (which got eval_loss=3.4866, ppl=32.67 on ckpt40000/1000 samples) -- if ckpt80000's
# L1024/500-sample number lands in a sane, comparable range (lower loss than ckpt40000,
# since it's further trained), the pipeline is confirmed fine for the L4096 run too.

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

OUTPUT="${BASE}/runs/gpt2_medium_owt_rotary_80k/ppl_quick_ckpt80000/L1024_sanity500"
mkdir -p "${OUTPUT}"
MASTER_PORT=$(( 13000 + SLURM_JOB_ID % 10000 ))

echo "=== rotary medium ckpt80000 ppl @ block_size=1024 (500 samples, sanity check) ==="

python -m torch.distributed.run --nproc_per_node=1 --master_port=${MASTER_PORT} ./run_clm.py \
  --model_type gpt2 --tokenizer_name gpt2 \
  --model_name_or_path "${CKPT}" \
  --dataset_name openwebtext \
  --validation_split_percentage 1 \
  --max_eval_samples 500 \
  --preprocessing_num_workers 8 \
  --pe_method rotary --attn_implementation eager \
  --block_size 1024 \
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

python3 -c "import json; d=json.load(open('${OUTPUT}/eval_results.json')); print(f'rotary medium ckpt80000 L1024 (500 samples, sanity): eval_loss={d[\"eval_loss\"]:.4f} ppl={d[\"perplexity\"]:.2f}')"

echo "=== Done: rotary medium ckpt80000 L1024 sanity check ==="

#!/bin/bash
#SBATCH --job-name=waveletpe_hp_alllen
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_waveletpe_s42_alllen.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:1
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Table 4 (GPT-2 small): matched-lineage "Wavelet PE" external baseline, ALL 6 lengths.
# This is the genuine pretrain=wavelet, finetune=wavelet checkpoint that ALiBi/RoPE/NoPE's shared
# wikitext_pe_cmp pipeline produces (NoPE's checkpoint deviates from this lineage at finetune time,
# see PAT-256) — never evaluated on HotpotQA-Long before (only xsum). Checkpoint config.json confirms:
# pe_method=wavelet, wavelet_router=False (single fixed wavelet bias, not the multi-scale QWAB router),
# wavelet_mode=logit_bias_ctxscale_shift_v0, attn_implementation=eager.
# Checkpoint: runs/wikitext_pe_cmp/wavelet/finetune_mix_seed42/checkpoint-15900

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
CKPT="${BASE}/runs/wikitext_pe_cmp/wavelet/finetune_mix_seed42/checkpoint-15900"
cd "${BASE}"

for BSIZE in 512 2048 4096 8192 12288 16384; do
  if [ "${BSIZE}" -le 4096 ]; then
    JSONL="${BASE}/hotpot_long/data/hotpot_long_dev_uniform.jsonl"
  else
    JSONL="${BASE}/hotpot_long/data/hotpot_long_dev_uniform_${BSIZE}only.jsonl"
  fi
  OUTPUT="${BASE}/hotpot_long/results_uniform/waveletpe_s42_ckpt15900/L${BSIZE}"
  mkdir -p "${OUTPUT}"
  echo "=== waveletpe s42 L${BSIZE} ==="
  MASTER_PORT=$(( 13000 + SLURM_JOB_ID % 10000 + BSIZE % 100 ))
  if [ "${BSIZE}" -gt 4096 ]; then
    PRECISION_ARGS="--bf16 True"
  else
    PRECISION_ARGS=""
  fi
  python -m torch.distributed.run --nproc_per_node=1 --master_port=${MASTER_PORT} ./run_clm.py \
    --model_type gpt2 --tokenizer_name gpt2 \
    --model_name_or_path "${CKPT}" \
    --pe_method wavelet --attn_implementation eager \
    --dataset_name hotpot_qa --dataset_config_name distractor \
    --hotpot_long_jsonl "${JSONL}" \
    --hotpot_long_lengths "${BSIZE}" \
    --do_eval --block_size "${BSIZE}" \
    --per_device_eval_batch_size 1 \
    ${PRECISION_ARGS} \
    --output_dir "${OUTPUT}" --overwrite_output_dir \
    --logging_dir "${OUTPUT}/log" \
    --seed 42 --load_best_model_at_end False \
    --share_freq_across_heads True --wavelet_router False \
    --wavelet_mode logit_bias_ctxscale_shift_v0 --scale_range 0 16 \
    --router_band_num 8 --use_beta_modulation False \
    --use_soft_wavelet_fox False --wavelet_baseline_use False \
    --single_A_B True --num_harmonics 1
  python3 -c "import json; d=json.load(open('${OUTPUT}/eval_results.json')); print(f'waveletpe s42 L${BSIZE}: F1={d[\"eval_f1\"]:.4f}')"
done

echo "=== Done: Wavelet PE s42 L512/2048/4096/8192/12288/16384 ==="

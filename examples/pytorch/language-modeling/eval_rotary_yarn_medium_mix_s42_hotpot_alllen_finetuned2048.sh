#!/bin/bash
#SBATCH --job-name=yarnft2048_hp_alllen
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_rotary_yarn_medium_s42_finetuned2048_alllen.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:2
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Full-length HotpotQA-Long F1 sweep for the GENUINELY YaRN-FINETUNED Rotary GPT-2-medium
# checkpoint (mix_medium_owt_rotary_yarn_2048_s42_fp32/checkpoint-400, trained at the
# target length 2048 with yarn_factor=4 per the YaRN paper's own protocol -- NOT the
# zero-shot eval done earlier in eval_rotary_yarn_medium_mix_s42_hotpot_alllen.sh, which
# reused the plain-Rotary checkpoint with only the frequency formula swapped at eval time
# and showed no benefit (L2048 F1=0.0822, L4096 F1=0.0216 -- both ~= plain RoPE).
# This checkpoint's own weights were actually adapted via 400 finetune steps at L2048, so
# this is the real test of whether YaRN's method (not just its frequency formula) helps.
# yarn_factor = target_length/512 recomputed at each length (matches the zero-shot script's
# convention) -- at L2048 this reduces to exactly the factor the model was tuned for; at
# other lengths it's an out-of-distribution extrapolation FROM the L2048 tuning point.
# Includes L512 (unlike the zero-shot sweep, which skipped it as a known no-op on the
# untouched plain checkpoint) since this checkpoint's weights differ from plain Rotary's --
# worth checking whether YaRN finetuning at L2048 costs anything at the original L512
# length (catastrophic forgetting check).
# Compare against: plain RoPE (eval_rotary_medium_mix_s42_hotpot_alllen.sh) and zero-shot
# YaRN (eval_rotary_yarn_medium_mix_s42_hotpot_alllen.sh) at the same lengths.

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
CKPT="${BASE}/runs/mix_medium_owt_rotary_yarn_2048_s42_fp32/checkpoint-400"
cd "${BASE}"

for BSIZE in 512 2048 4096 8192 12288 16384; do
  if [ "${BSIZE}" -le 4096 ]; then
    JSONL="${BASE}/hotpot_long/data/hotpot_long_dev.jsonl"
  else
    JSONL="${BASE}/hotpot_long/data/hotpot_long_dev_uniform_${BSIZE}only.jsonl"
  fi
  YARN_FACTOR=$(python3 -c "print(${BSIZE}/512)")
  OUTPUT="${BASE}/hotpot_long/results/rotary_yarn_medium_s42_finetuned2048_ckpt400/L${BSIZE}"
  mkdir -p "${OUTPUT}/log"
  echo "=== Rotary+YaRN medium (genuinely finetuned @2048) s42 HotpotQA-Long L${BSIZE} (yarn_factor=${YARN_FACTOR}) ==="
  MASTER_PORT=$(( 15000 + SLURM_JOB_ID % 10000 + BSIZE % 100 ))
  python -m torch.distributed.run --nproc_per_node=2 --master_port=${MASTER_PORT} ./run_clm.py \
    --model_type gpt2 --tokenizer_name gpt2 \
    --model_name_or_path "${CKPT}" \
    --attn_implementation eager \
    --pe_method rotary \
    --use_yarn True \
    --yarn_factor "${YARN_FACTOR}" \
    --yarn_original_max_position_embeddings 512 \
    --dataset_name hotpot_qa --dataset_config_name distractor \
    --hotpot_long_jsonl "${JSONL}" \
    --hotpot_long_lengths "${BSIZE}" \
    --do_eval \
    --block_size "${BSIZE}" \
    --per_device_eval_batch_size 1 \
    --output_dir "${OUTPUT}" --overwrite_output_dir \
    --logging_dir "${OUTPUT}/log" \
    --seed 42 --load_best_model_at_end False
  python3 -c "import json; d=json.load(open('${OUTPUT}/eval_results.json')); print(f'Rotary+YaRN medium (finetuned@2048) s42 L${BSIZE}: F1={d[\"eval_f1\"]:.4f} eval_loss={d[\"eval_loss\"]:.4f}')"
done

echo "=== Done: Rotary+YaRN medium (genuinely finetuned @2048) s42 HotpotQA-Long full-length sweep ==="

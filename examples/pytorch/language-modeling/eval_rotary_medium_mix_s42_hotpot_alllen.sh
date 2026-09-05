#!/bin/bash
#SBATCH --job-name=rotmed_hp_alllen
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_rotary_medium_s42_alllen.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:2
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Full-length HotpotQA-Long F1 sweep for the FINETUNED Rotary GPT-2-medium checkpoint
# (mix_medium_owt_rotary_10ep_s42_fp32/checkpoint-15000), matching Table 4's full length
# set (L512/2048/4096/8192/12288/16384). L4096 already done separately (job 577446:
# F1=0.0246, eval_loss=9.42 -- catastrophic 8x-context extrapolation collapse, consistent
# with known RoPE behavior). This script fills in the remaining 5 lengths.
# L512/L2048: front-placed hotpot_long_dev.jsonl (same file as L4096's row, Table4 base
# convention). L8192/12288/16384: per-length hotpot_long_dev_uniform_{L}only.jsonl --
# despite the "_uniform_" filename, the actual assembled context is still front-pinned
# because build_context_budgeted's default (respect_doc_order=False) always puts
# supporting-fact sentences first regardless of the source file's placement label (see
# PAT-253 session notes) -- this matches the small-model extralen scripts' precedent
# exactly (e.g. eval_rotary_s42_extralen.sh).
# fp32 throughout (matching the checkpoint's own training precision) for L512/2048/8192;
# --bf16 True added for L12288/16384 if fp32 OOMs at medium scale (24 layers, 1024 hidden,
# eager attention -- untested at these lengths for THIS model size; small model needed
# similar precision/memory workarounds at L8192+). Watch the log for OOM and adjust if
# it happens -- this is not pre-verified to fit.

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
CKPT="${BASE}/runs/mix_medium_owt_rotary_10ep_s42_fp32/checkpoint-15000"
cd "${BASE}"

for BSIZE in 512 2048 8192 12288 16384; do
  if [ "${BSIZE}" -le 4096 ]; then
    JSONL="${BASE}/hotpot_long/data/hotpot_long_dev.jsonl"
  else
    JSONL="${BASE}/hotpot_long/data/hotpot_long_dev_uniform_${BSIZE}only.jsonl"
  fi
  OUTPUT="${BASE}/hotpot_long/results/rotary_medium_s42_ckpt15000/L${BSIZE}"
  mkdir -p "${OUTPUT}/log"
  echo "=== Rotary medium (finetuned) s42 HotpotQA-Long L${BSIZE} ==="
  MASTER_PORT=$(( 13000 + SLURM_JOB_ID % 10000 + BSIZE % 100 ))
  python -m torch.distributed.run --nproc_per_node=2 --master_port=${MASTER_PORT} ./run_clm.py \
    --model_type gpt2 --tokenizer_name gpt2 \
    --model_name_or_path "${CKPT}" \
    --attn_implementation eager \
    --pe_method rotary \
    --dataset_name hotpot_qa --dataset_config_name distractor \
    --hotpot_long_jsonl "${JSONL}" \
    --hotpot_long_lengths "${BSIZE}" \
    --do_eval \
    --block_size "${BSIZE}" \
    --per_device_eval_batch_size 1 \
    --output_dir "${OUTPUT}" --overwrite_output_dir \
    --logging_dir "${OUTPUT}/log" \
    --seed 42 --load_best_model_at_end False
  python3 -c "import json; d=json.load(open('${OUTPUT}/eval_results.json')); print(f'Rotary medium (finetuned) s42 L${BSIZE}: F1={d[\"eval_f1\"]:.4f} eval_loss={d[\"eval_loss\"]:.4f}')"
done

echo "=== Done: Rotary medium (finetuned) s42 HotpotQA-Long full-length sweep ==="

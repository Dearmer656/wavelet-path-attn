#!/bin/bash
#SBATCH --job-name=alibimed_hp_alllen
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_alibi_medium_s42_alllen.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:2
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Full-length HotpotQA-Long F1 sweep for the FINETUNED ALiBi GPT-2-medium checkpoint
# (mix_medium_owt_alibi_10ep_s42_fp32/checkpoint-15000), matching Table 4's full length
# set (L512/2048/4096/8192/12288/16384). checkpoint-15000 assumed by analogy with the
# Rotary counterpart (train_gpt2_medium_owt_mix_rotary_10ep_s42_fp32.sh -> checkpoint-15000)
# since both finetunes use the identical "mix" dataset, 10 epochs, global_bs=64 -- if the
# actual final checkpoint differs, fix CKPT below before relying on these numbers.
# ALiBi needs no YaRN/RoPE-frequency reconfiguration at eval time -- its whole design
# point is length generalization without retraining, so this is a plain zero-shot sweep
# on the same finetuned weights across all lengths (unlike Rotary, which needs YaRN or
# collapses catastrophically beyond training length).
# L512/L2048/L4096: front-placed hotpot_long_dev.jsonl. L8192/12288/16384: per-length
# hotpot_long_dev_uniform_{L}only.jsonl (still front-pinned in practice -- see PAT-253
# session notes on build_context_budgeted's respect_doc_order default).
# eager attention throughout for eval (the finetune itself now uses flash_attention_2,
# 2026-09-06 update to match QWAB's own finetune convention -- eval stays on eager since
# this project already verified ALiBi's eager vs flash_attention_2 are numerically
# equivalent, and eager avoids any flash-attention edge cases at very long eval lengths).

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
CKPT="${BASE}/runs/mix_medium_owt_alibi_10ep_s42_fp32/checkpoint-15000"
cd "${BASE}"

for BSIZE in 512 2048 4096 8192 12288 16384; do
  if [ "${BSIZE}" -le 4096 ]; then
    JSONL="${BASE}/hotpot_long/data/hotpot_long_dev.jsonl"
  else
    JSONL="${BASE}/hotpot_long/data/hotpot_long_dev_uniform_${BSIZE}only.jsonl"
  fi
  OUTPUT="${BASE}/hotpot_long/results/alibi_medium_s42_ckpt15000/L${BSIZE}"
  mkdir -p "${OUTPUT}/log"
  echo "=== ALiBi medium (finetuned) s42 HotpotQA-Long L${BSIZE} ==="
  MASTER_PORT=$(( 14000 + SLURM_JOB_ID % 10000 + BSIZE % 100 ))
  python -m torch.distributed.run --nproc_per_node=2 --master_port=${MASTER_PORT} ./run_clm.py \
    --model_type gpt2 --tokenizer_name gpt2 \
    --model_name_or_path "${CKPT}" \
    --attn_implementation eager \
    --pe_method alibi \
    --dataset_name hotpot_qa --dataset_config_name distractor \
    --hotpot_long_jsonl "${JSONL}" \
    --hotpot_long_lengths "${BSIZE}" \
    --do_eval \
    --block_size "${BSIZE}" \
    --per_device_eval_batch_size 1 \
    --output_dir "${OUTPUT}" --overwrite_output_dir \
    --logging_dir "${OUTPUT}/log" \
    --seed 42 --load_best_model_at_end False
  python3 -c "import json; d=json.load(open('${OUTPUT}/eval_results.json')); print(f'ALiBi medium (finetuned) s42 L${BSIZE}: F1={d[\"eval_f1\"]:.4f} eval_loss={d[\"eval_loss\"]:.4f}')"
done

echo "=== Done: ALiBi medium (finetuned) s42 HotpotQA-Long full-length sweep ==="

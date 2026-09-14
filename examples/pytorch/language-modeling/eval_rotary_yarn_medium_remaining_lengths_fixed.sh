#!/bin/bash
#SBATCH --job-name=rotyarn_remaining_fixed
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/log_file/train/%j_rotary_yarn_medium_s42_remaining_fixed.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:p6000:2
#SBATCH --time=25:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Zero-shot YaRN eval at L4096/8192/12288/16384, BUGFIXED version (L2048 already
# submitted separately as job 588952). Fix: run_clm.py now reapplies
# _apply_yarn_to_rotary_embedding right after from_pretrained returns, since
# rotary_embedding_torch's `freqs` is an nn.Parameter (part of state_dict) that
# from_pretrained's checkpoint load was silently reverting to plain-RoPE values
# every time -- verified directly, every prior YaRN result (zero-shot AND the
# 400-step finetuned versions) never actually used YaRN's frequency rescale.
# Old (buggy) zero-shot numbers for reference: L4096=0.0216, L8192/12288/16384
# never even ran under the zero-shot condition (only the finetuned/unfair
# checkpoints were evaluated at those lengths).
# 2x p6000 + per_device_eval_batch_size=4 (not the 3090x2/bs=1 used for L2048,
# job 588952) -- p6000's larger memory lets this finish in minutes instead of
# ~30min per length. Held back from submission until L2048 (job 588952)
# confirms the fix produces a sane number, per explicit request.

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

for BSIZE in 4096 8192 12288 16384; do
  if [ "${BSIZE}" -le 4096 ]; then
    JSONL="${BASE}/hotpot_long/data/hotpot_long_dev.jsonl"
  else
    if [ "${BSIZE}" -eq 16384 ]; then
      JSONL="${BASE}/hotpot_long/data/hotpot_long_dev_uniform_16384_large_pool.jsonl"
    else
      JSONL="${BASE}/hotpot_long/data/hotpot_long_dev_uniform_${BSIZE}only.jsonl"
    fi
  fi
  YARN_FACTOR=$(python3 -c "print(${BSIZE}/512)")
  OUTPUT="${BASE}/hotpot_long/results/rotary_yarn_medium_s42_ckpt15000_fixed/L${BSIZE}"
  mkdir -p "${OUTPUT}/log"
  echo "=== Rotary+YaRN medium (zero-shot, BUGFIXED) s42 HotpotQA-Long L${BSIZE} (yarn_factor=${YARN_FACTOR}) ==="
  MASTER_PORT=$(( 27500 + SLURM_JOB_ID % 1000 + BSIZE % 100 ))
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
    --per_device_eval_batch_size 4 \
    --output_dir "${OUTPUT}" --overwrite_output_dir \
    --logging_dir "${OUTPUT}/log" \
    --seed 42 --load_best_model_at_end False
  python3 -c "import json; d=json.load(open('${OUTPUT}/eval_results.json')); print(f'Rotary+YaRN medium (zero-shot, FIXED) s42 L${BSIZE}: F1={d[\"eval_f1\"]:.4f} eval_loss={d[\"eval_loss\"]:.4f}')"
done

echo "=== Done: Rotary+YaRN medium (zero-shot, FIXED) s42 L4096-16384 ==="

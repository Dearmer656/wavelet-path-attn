#!/bin/bash
#SBATCH --job-name=alibimed_hp_L16384
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_alibi_medium_s42_L16384_standalone.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:p6000:2
#SBATCH --nodelist=elm82
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Standalone L16384 HotpotQA-Long eval for ALiBi medium, run in parallel on separate GPUs
# while eval_alibi_medium_mix_s42_hotpot_alllen.sh (578004) is still working through
# L12288 sequentially -- same checkpoint-15000 as that script (not the true final
# checkpoint-15900) for consistency with the other 5 already-landed lengths in that same
# sweep.
# 2026-09-07: eager+bf16 (578723, and 578804 with --attn_implementation flash_attention_2
# on the CLI) OOM'd anyway with the exact same "Tried to allocate 16.00 GiB" -- traced this
# to a real bug: run_clm.py sets `config.attn_implementation` (public attribute) instead of
# the actual `_attn_implementation` property HF reads, so the CLI flag never took effect;
# AutoModelForCausalLM.from_pretrained/from_config then auto-resolves the unset
# _attn_implementation to 'sdpa', which ALiBi's own forward code treats as "not exactly
# flash_attention_2" and force-routes to eager regardless. Verified via a standalone
# diagnostic (not assumed) -- confirmed this also means ALiBi's own pretrain (checkpoint-
# 80000) and finetune (checkpoint-15000/15900) have been running eager the whole time
# despite being labeled flash_attention_2 throughout the project; existing F1/ppl numbers
# are still valid (eager computes the same result, just slower), only the "flash" label
# was wrong.
# Fix (background, no longer needed for THIS run): added "_attn_implementation" to
# force_override_hf_config's prefix whitelist (run_clm.py, ~line 4474) -- a minimal,
# additive, backward-compatible change. flash_attention_2's custom ALiBi kernel then hit
# its own separate wall ("only supports a pure causal mask" -- rejects the padding this
# batch needs), a dead end for this specific case.
# 2026-09-07 (later): switched to eager on elm82's p6000 (96GB/GPU, not the standard
# Quadro P6000's 24GB -- confirmed by the user) instead of a6000 (48GB, where eager OOM'd
# needing ~50.7GB total). Also capped --max_eval_samples 2000 to bound wall-clock time
# (does not affect the per-sample OOM risk, since each L16384 example needs the same
# memory regardless of how many total examples are evaluated -- this is purely a
# time-control knob, orthogonal to the memory fix).

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

BSIZE=16384
JSONL="${BASE}/hotpot_long/data/hotpot_long_dev_uniform_${BSIZE}only.jsonl"
OUTPUT="${BASE}/hotpot_long/results/alibi_medium_s42_ckpt15000/L${BSIZE}"
mkdir -p "${OUTPUT}/log"
echo "=== ALiBi medium (finetuned) s42 HotpotQA-Long L${BSIZE} (standalone, 2000 cases) ==="
MASTER_PORT=$(( 14500 + SLURM_JOB_ID % 10000 ))
python -m torch.distributed.run --nproc_per_node=2 --master_port=${MASTER_PORT} ./run_clm.py \
  --model_type gpt2 --tokenizer_name gpt2 \
  --model_name_or_path "${CKPT}" \
  --attn_implementation eager \
  --pe_method alibi \
  --bf16 True \
  --dataset_name hotpot_qa --dataset_config_name distractor \
  --hotpot_long_jsonl "${JSONL}" \
  --hotpot_long_lengths "${BSIZE}" \
  --do_eval \
  --max_eval_samples 2000 \
  --block_size "${BSIZE}" \
  --per_device_eval_batch_size 1 \
  --output_dir "${OUTPUT}" --overwrite_output_dir \
  --logging_dir "${OUTPUT}/log" \
  --ddp_timeout 21600 --seed 42 --load_best_model_at_end False
python3 -c "import json; d=json.load(open('${OUTPUT}/eval_results.json')); print(f'ALiBi medium (finetuned) s42 L${BSIZE}: F1={d[\"eval_f1\"]:.4f} eval_loss={d[\"eval_loss\"]:.4f}')"

echo "=== Done: ALiBi medium (finetuned) s42 HotpotQA-Long L16384 (standalone) ==="

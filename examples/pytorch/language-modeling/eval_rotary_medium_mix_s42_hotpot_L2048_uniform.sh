#!/bin/bash
#SBATCH --job-name=rotmed_hp_L2048_uniform
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_rotary_medium_s42_L2048_uniform.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:2
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# HotpotQA-Long F1 eval for the FINETUNED Rotary GPT-2-medium checkpoint at L1024, using
# the UNIFORM-placement dev set (hotpot_long_dev_uniform.jsonl) instead of the front-placed
# one used everywhere else this session. Purpose: test whether RoPE's near-zero F1 collapse
# under front-placement (L2048=0.0805, L4096=0.0246, L8192=0.0198) is specifically a
# consequence of front-placement always maximizing the evidence-to-query relative distance
# (RoPE's worst extrapolation regime) -- under uniform placement, some fraction of examples
# should have evidence near the end (small relative distance, still in-range for RoPE),
# so a materially higher F1 here than the front-placed number would confirm that
# hypothesis rather than a pipeline bug.
# Checkpoint: runs/mix_medium_owt_rotary_10ep_s42_fp32/checkpoint-15000 (the finetune).

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
JSONL="${BASE}/hotpot_long/data/hotpot_long_dev_uniform.jsonl"
OUTPUT="${BASE}/hotpot_long/results/rotary_medium_s42_ckpt15000/L2048_uniform"
mkdir -p "${OUTPUT}"
cd "${BASE}"

MASTER_PORT=$(( 13000 + SLURM_JOB_ID % 10000 ))

echo "=== Rotary medium (finetuned) s42 HotpotQA-Long L2048_uniform ==="

python -m torch.distributed.run --nproc_per_node=2 --master_port=${MASTER_PORT} ./run_clm.py \
  --model_type gpt2 --tokenizer_name gpt2 \
  --model_name_or_path "${CKPT}" \
  --attn_implementation eager \
  --pe_method rotary \
  --dataset_name hotpot_qa --dataset_config_name distractor \
  --hotpot_long_jsonl "${JSONL}" \
  --hotpot_long_lengths 2048 \
  --do_eval \
  --block_size 2048 \
  --per_device_eval_batch_size 1 \
  --output_dir "${OUTPUT}" --overwrite_output_dir \
  --logging_dir "${OUTPUT}/log" \
  --seed 42 --load_best_model_at_end False

python3 -c "import json; d=json.load(open('${OUTPUT}/eval_results.json')); print(f'Rotary medium (finetuned) s42 L2048_uniform: F1={d[\"eval_f1\"]:.4f}')"

echo "=== Done: Rotary medium (finetuned) s42 HotpotQA-Long L2048_uniform ==="

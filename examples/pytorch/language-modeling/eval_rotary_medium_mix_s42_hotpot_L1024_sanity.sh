#!/bin/bash
#SBATCH --job-name=rotmed_hp_L1024_san
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_rotary_medium_s42_L1024.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:2
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# HotpotQA-Long F1 eval for the FINETUNED (not pretrain) plain Rotary GPT-2-medium
# checkpoint at L1024, matching Table 4's small-model methodology 1:1 (front-placed
# hotpot_long_dev.jsonl -- the paper's chosen convention, NOT hotpot_long_dev_uniform.jsonl
# -- same file/flags as eval_rotary_s42_extralen.sh's small-model L8192+ rows, just at the
# medium checkpoint and L1024).
# Checkpoint: runs/mix_medium_owt_rotary_10ep_s42_fp32/checkpoint-15000 (the mix finetune,
# NOT gpt2_medium_owt_rotary_80k/checkpoint-80000's plain OWT pretrain -- corrected after
# initially running the wrong checkpoint, see PAT-164/PAT-253 session notes).

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
JSONL="${BASE}/hotpot_long/data/hotpot_long_dev.jsonl"
OUTPUT="${BASE}/hotpot_long/results/rotary_medium_s42_ckpt15000/L1024_sanity500"
mkdir -p "${OUTPUT}"
cd "${BASE}"

MASTER_PORT=$(( 13000 + SLURM_JOB_ID % 10000 ))

echo "=== Rotary medium (finetuned) s42 HotpotQA-Long L1024 ==="

python -m torch.distributed.run --nproc_per_node=2 --master_port=${MASTER_PORT} ./run_clm.py \
  --model_type gpt2 --tokenizer_name gpt2 \
  --model_name_or_path "${CKPT}" \
  --attn_implementation eager \
  --pe_method rotary \
  --dataset_name hotpot_qa --dataset_config_name distractor \
  --hotpot_long_jsonl "${JSONL}" \
  --hotpot_long_lengths 1024 \
  --do_eval \
  --block_size 1024 --max_eval_samples 500 \
  --per_device_eval_batch_size 1 \
  --output_dir "${OUTPUT}" --overwrite_output_dir \
  --logging_dir "${OUTPUT}/log" \
  --seed 42 --load_best_model_at_end False

python3 -c "import json; d=json.load(open('${OUTPUT}/eval_results.json')); print(f'Rotary medium (finetuned) s42 L1024: F1={d[\"eval_f1\"]:.4f}')"

echo "=== Done: Rotary medium (finetuned) s42 HotpotQA-Long L1024 ==="

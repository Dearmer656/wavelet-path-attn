#!/bin/bash
#SBATCH --job-name=rotary_ntk_medium_L2048
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/log_file/train/%j_rotary_ntk_medium_L2048_fixed.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:p6000:3
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Controlled comparison for the "YaRN at L2048 (medium) showed no F1 benefit, but
# NTK at L1536 (small) showed a real one -- is this a YaRN-implementation problem
# or just 'medium+4x is too hard for any zero-shot RoPE trick'?" question. Runs
# plain NTK-aware theta scaling (not YaRN) on the SAME medium checkpoint/length
# (L2048, factor=4) that YaRN failed to help at, using the standard NTK-aware
# formula theta_new = theta_base * factor^(head_dim/(head_dim-2)):
# theta = 10000 * 4^(64/62) = 41829.37. Uses the same RoPE checkpoint-load bugfix
# (already generalized to plain --rope_theta, not just --use_yarn).

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

BSIZE=2048
JSONL="${BASE}/hotpot_long/data/hotpot_long_dev.jsonl"
OUTPUT="${BASE}/hotpot_long/results/rotary_ntk_medium_s42_ckpt15000_fixed/L${BSIZE}"
mkdir -p "${OUTPUT}/log"
echo "=== Rotary NTK (BUGFIXED, medium) s42 HotpotQA-Long L${BSIZE} theta=41829.37 ==="
MASTER_PORT=$(( 30500 + SLURM_JOB_ID % 1000 ))
python -m torch.distributed.run --nproc_per_node=3 --master_port=${MASTER_PORT} ./run_clm.py \
  --model_type gpt2 --tokenizer_name gpt2 \
  --model_name_or_path "${CKPT}" \
  --attn_implementation eager \
  --pe_method rotary \
  --rope_theta 41829.37 \
  --dataset_name hotpot_qa --dataset_config_name distractor \
  --hotpot_long_jsonl "${JSONL}" \
  --hotpot_long_lengths "${BSIZE}" \
  --do_eval \
  --block_size "${BSIZE}" \
  --per_device_eval_batch_size 4 \
  --output_dir "${OUTPUT}" --overwrite_output_dir \
  --logging_dir "${OUTPUT}/log" \
  --seed 42 --load_best_model_at_end False
python3 -c "import json; d=json.load(open('${OUTPUT}/eval_results.json')); print(f'Rotary NTK (medium, FIXED) s42 L${BSIZE}: F1={d[\"eval_f1\"]:.4f} EM={d[\"eval_em\"]:.4f} eval_loss={d[\"eval_loss\"]:.4f}')"

echo "=== Done: Rotary NTK (medium, FIXED) s42 L2048 ==="

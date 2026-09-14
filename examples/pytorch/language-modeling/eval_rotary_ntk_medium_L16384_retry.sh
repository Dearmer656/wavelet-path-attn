#!/bin/bash
#SBATCH --job-name=rotary_ntk_medium_L16384
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/log_file/train/%j_rotary_ntk_medium_L16384_retry.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:p6000:3
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Retry of L16384 NTK-aware eval (medium) -- job 588996's bs=4 attempt OOM'd
# (tried to allocate 64GiB, ~95GB card already at 68.63GB used). Eager attention
# at L=16384 for a 24L/16H model needs much more memory per example than shorter
# lengths; dropping to bs=1 here.

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

BSIZE=16384
THETA=357851.7978
JSONL="${BASE}/hotpot_long/data/hotpot_long_dev_uniform_16384_large_pool.jsonl"
OUTPUT="${BASE}/hotpot_long/results/rotary_ntk_medium_s42_ckpt15000_fixed/L${BSIZE}"
mkdir -p "${OUTPUT}/log"
echo "=== Rotary NTK (medium, FIXED) s42 HotpotQA-Long L${BSIZE} theta=${THETA}, bs=1 ==="
MASTER_PORT=$(( 33500 + SLURM_JOB_ID % 1000 ))
python -m torch.distributed.run --nproc_per_node=3 --master_port=${MASTER_PORT} ./run_clm.py \
  --model_type gpt2 --tokenizer_name gpt2 \
  --model_name_or_path "${CKPT}" \
  --attn_implementation eager \
  --pe_method rotary \
  --rope_theta "${THETA}" \
  --dataset_name hotpot_qa --dataset_config_name distractor \
  --hotpot_long_jsonl "${JSONL}" \
  --hotpot_long_lengths "${BSIZE}" \
  --do_eval \
  --block_size "${BSIZE}" \
  --per_device_eval_batch_size 1 \
  --output_dir "${OUTPUT}" --overwrite_output_dir \
  --logging_dir "${OUTPUT}/log" \
  --seed 42 --load_best_model_at_end False
python3 -c "import json; d=json.load(open('${OUTPUT}/eval_results.json')); print(f'Rotary NTK (medium, FIXED) s42 L${BSIZE}: F1={d[\"eval_f1\"]:.4f} EM={d[\"eval_em\"]:.4f} eval_loss={d[\"eval_loss\"]:.4f}')"

echo "=== Done ==="

#!/bin/bash
#SBATCH --job-name=rotary_ntk_medium_remaining
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/log_file/train/%j_rotary_ntk_medium_remaining.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:p6000:3
#SBATCH --time=25:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Zero-shot NTK-aware theta scaling, medium model, L4096/8192/12288/16384
# (L2048 already done: theta=41829.37 -> F1=0.5875, job 588970). Same standard
# formula as this project's own eval_rotary_ntk_xsum_s42.sh:
# theta_new = 10000 * (L/512)^(head_dim/(head_dim-2)), head_dim=64.
# Uses the RoPE checkpoint-load bugfix (generalized to plain --rope_theta, not
# just --use_yarn) -- verified against a historical ground-truth reproduction
# (exact match to a pre-regression 2026-05 NTK result).
# This is the fair (no extra target-length training), zero-shot, project-
# standard-methodology NTK row intended for the paper's length-extrapolation
# baseline table, replacing YaRN (which underperformed plain NTK badly at the
# same L2048/factor=4 point -- 0.0822 vs 0.5875 -- traced to YaRN's NTK-by-parts
# ramp leaving high-frequency dims under-rescaled for this model/factor combo,
# not a remaining bug).

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

declare -A THETAS=( [4096]=85550.3759 [8192]=174969.5854 [12288]=265909.7030 [16384]=357851.7978 )

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
  THETA="${THETAS[$BSIZE]}"
  OUTPUT="${BASE}/hotpot_long/results/rotary_ntk_medium_s42_ckpt15000_fixed/L${BSIZE}"
  mkdir -p "${OUTPUT}/log"
  echo "=== Rotary NTK (medium, FIXED) s42 HotpotQA-Long L${BSIZE} theta=${THETA} ==="
  MASTER_PORT=$(( 32500 + SLURM_JOB_ID % 1000 + BSIZE % 100 ))
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
    --per_device_eval_batch_size 4 \
    --output_dir "${OUTPUT}" --overwrite_output_dir \
    --logging_dir "${OUTPUT}/log" \
    --seed 42 --load_best_model_at_end False
  python3 -c "import json; d=json.load(open('${OUTPUT}/eval_results.json')); print(f'Rotary NTK (medium, FIXED) s42 L${BSIZE}: F1={d[\"eval_f1\"]:.4f} EM={d[\"eval_em\"]:.4f} eval_loss={d[\"eval_loss\"]:.4f}')"
done

echo "=== Done: Rotary NTK (medium, FIXED) s42 L4096-16384 ==="

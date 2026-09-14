#!/bin/bash
#SBATCH --job-name=rotary_ntk_xsum_L1536_fixed
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/log_file/train/%j_rotary_ntk_xsum_s42_L1536_fixed.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:p6000:2
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Rerun of eval_rotary_ntk_xsum_s42.sh's L1536 case (small, 12L/12H model) after
# fixing the RoPE checkpoint-load bug (freqs silently reverted by from_pretrained,
# same root cause as the YaRN bug). Historical (May 2026, pre-regression) result:
# rougeL=0.2673 vs plain rope_theta=10000's 0.1731 -- a real, large improvement.
# Directly verified via forward-pass diff that the CURRENT (pre-fix) code gives
# bit-identical logits regardless of rope_theta on this exact checkpoint; this run
# checks whether the fix restores that historical-magnitude improvement.

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true
export SKIP_FENICE=1
export SKIP_SUMMAC=1

BASE=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
CKPT="${BASE}/runs/rotary_mix_finetune/s42/checkpoint-15000"
XSUM_FILE="/cl/work5/hongyu-s/fact-check-summarization/xsum_test_filter_level2_official_style.jsonl"
OUT_DIR="${BASE}/runs/rotary_mix_finetune/s42/ckpt_eval_xsum_rouge_ntk_fixed/xsum_L1536"
mkdir -p "${OUT_DIR}"
cd "${BASE}"

echo "=== Rotary NTK (BUGFIXED) XSum L1536 theta=31074 ==="
MASTER_PORT=$(( 29500 + SLURM_JOB_ID % 1000 ))
python -m torch.distributed.run --nproc_per_node=2 --master_port=${MASTER_PORT} ./run_clm.py \
  --model_type gpt2 --tokenizer_name gpt2 \
  --model_name_or_path "${CKPT}" \
  --attn_implementation eager \
  --pe_method rotary \
  --rope_theta 31074 \
  --dataset_name xsum \
  --dataset_config_name default \
  --validation_file "${XSUM_FILE}" \
  --do_eval \
  --block_size 1536 \
  --per_device_eval_batch_size 4 \
  --xsum_bucket_size 512 \
  --xsum_bucket_apply_to eval_test \
  --output_dir "${OUT_DIR}" --overwrite_output_dir \
  --logging_dir "${OUT_DIR}/log" \
  --load_best_model_at_end False \
  --seed 42
python3 -c "import json; d=json.load(open('${OUT_DIR}/eval_results.json')); print(f'Rotary NTK (FIXED) L1536: rouge1={d[\"eval_rouge1\"]:.4f} rougeL={d[\"eval_rougeL\"]:.4f} eval_loss={d[\"eval_loss\"]:.4f}')"

echo "=== Done: Rotary NTK (FIXED) s42 XSum L1536 ==="

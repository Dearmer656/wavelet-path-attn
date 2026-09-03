#!/bin/bash
#SBATCH --job-name=rotary_ntk_extralen
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_rotary_ntk_s42_extralen.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:4
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Table 4 (GPT-2 small) fill-in: RoPE-NTK at L8192/12288/16384
# Same checkpoint as rotary_pe (RoPE and RoPE-NTK differ only in eval-time rope_theta)
# NTK formula: theta_new = 10000 * (L/512)^(64/62)
# L=8192: 174970  L=12288: 265910  L=16384: 357852

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true

BASE=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
CKPT="${BASE}/runs/rotary_mix_finetune/s42/checkpoint-15000"
cd "${BASE}"

declare -a LENGTHS=(8192   12288   16384)
declare -a THETAS=(174970  265910  357852)

for i in "${!LENGTHS[@]}"; do
  L="${LENGTHS[$i]}"
  THETA="${THETAS[$i]}"
  JSONL="${BASE}/hotpot_long/data/hotpot_long_dev_uniform_${L}only.jsonl"
  OUTPUT="${BASE}/hotpot_long/results_uniform/rotary_ntk_s42_ckpt15000_fixed/L${L}"
  mkdir -p "${OUTPUT}"
  echo "=== Rotary NTK (fixed) s42 L${L} theta=${THETA} ==="
  MASTER_PORT=$(( 17000 + SLURM_JOB_ID % 1000 + L % 100 ))
  python -m torch.distributed.run --nproc_per_node=4 --master_port=${MASTER_PORT} ./run_clm.py \
    --model_type gpt2 --tokenizer_name gpt2 \
    --model_name_or_path "${CKPT}" \
    --attn_implementation eager \
    --pe_method rotary \
    --rope_theta "${THETA}" \
    --dataset_name hotpot_qa --dataset_config_name distractor \
    --hotpot_long_jsonl "${JSONL}" \
    --hotpot_long_lengths "${L}" \
    --do_eval \
    --block_size "${L}" \
    --per_device_eval_batch_size 1 \
    --output_dir "${OUTPUT}" --overwrite_output_dir \
    --logging_dir "${OUTPUT}/log" \
    --seed 42 --load_best_model_at_end False
  python3 -c "import json; d=json.load(open('${OUTPUT}/eval_results.json')); print(f'Rotary NTK (fixed) L${L}: F1={d[\"eval_f1\"]:.4f} EM={d[\"eval_em\"]:.4f}')"
done

echo "=== Done: Rotary NTK (fixed) s42 L8192/12288/16384 ==="

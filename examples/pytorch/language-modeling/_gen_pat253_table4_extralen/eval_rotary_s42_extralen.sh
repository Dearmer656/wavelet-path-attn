#!/bin/bash
#SBATCH --job-name=rotary_hp_extralen
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_rotary_s42_extralen.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:4
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Table 4 (GPT-2 small) fill-in: plain RoPE (base theta=10000) at L8192/12288/16384
# Same checkpoint as L512/2048/4096 rows: runs/rotary_mix_finetune/s42/checkpoint-15000

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets

BASE=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
CKPT="${BASE}/runs/rotary_mix_finetune/s42/checkpoint-15000"
cd "${BASE}"

for BSIZE in 8192 12288 16384; do
  JSONL="${BASE}/hotpot_long/data/hotpot_long_dev_uniform_${BSIZE}only.jsonl"
  OUTPUT="${BASE}/hotpot_long/results_uniform/rotary_pe_s42_ckpt15000_fix/L${BSIZE}"
  mkdir -p "${OUTPUT}"
  echo "=== rotary_pe s42 L${BSIZE} ==="
  MASTER_PORT=$(( 13000 + SLURM_JOB_ID % 10000 + BSIZE % 100 ))
  python -m torch.distributed.run --nproc_per_node=4 --master_port=${MASTER_PORT} ./run_clm.py \
    --model_type gpt2 --tokenizer_name gpt2 \
    --model_name_or_path "${CKPT}" \
    --attn_implementation eager \
    --pe_method rotary \
    --dataset_name hotpot_qa --dataset_config_name distractor \
    --hotpot_long_jsonl "${JSONL}" \
    --hotpot_long_lengths "${BSIZE}" \
    --do_eval \
    --block_size "${BSIZE}" \
    --per_device_eval_batch_size 1 \
    --output_dir "${OUTPUT}" --overwrite_output_dir \
    --logging_dir "${OUTPUT}/log" \
    --seed 42 --load_best_model_at_end False
  python3 -c "import json; d=json.load(open('${OUTPUT}/eval_results.json')); print(f'rotary_pe s42 L${BSIZE}: F1={d[\"eval_f1\"]:.4f}')"
done

echo "=== Done: RoPE (plain) s42 L8192/12288/16384 ==="

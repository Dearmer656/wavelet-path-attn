#!/bin/bash
#SBATCH --job-name=rotary_ntk_L16384
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_rotary_ntk_s42_L16384.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:2
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# RoPE-NTK L16384 only — L8192/12288 already succeeded on 3090x2 (24GB), L16384 OOM'd there
# even with bf16+expandable_segments (3090 has half the memory of a6000/6000). Retry on 6000x2 (48GB).

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
CKPT="${BASE}/runs/rotary_mix_finetune/s42/checkpoint-15000"
cd "${BASE}"

L=16384
THETA=357852
JSONL="${BASE}/hotpot_long/data/hotpot_long_dev_uniform_${L}only.jsonl"
OUTPUT="${BASE}/hotpot_long/results_uniform/rotary_ntk_s42_ckpt15000_fixed/L${L}"
mkdir -p "${OUTPUT}"
echo "=== Rotary NTK (fixed) s42 L${L} theta=${THETA} ==="
MASTER_PORT=$(( 17000 + SLURM_JOB_ID % 1000 ))
python -m torch.distributed.run --nproc_per_node=2 --master_port=${MASTER_PORT} ./run_clm.py \
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
  --bf16 True \
  --output_dir "${OUTPUT}" --overwrite_output_dir \
  --logging_dir "${OUTPUT}/log" \
  --seed 42 --load_best_model_at_end False
python3 -c "import json; d=json.load(open('${OUTPUT}/eval_results.json')); print(f'Rotary NTK (fixed) L${L}: F1={d[\"eval_f1\"]:.4f} EM={d[\"eval_em\"]:.4f}')"

echo "=== Done: Rotary NTK (fixed) s42 L16384 ==="

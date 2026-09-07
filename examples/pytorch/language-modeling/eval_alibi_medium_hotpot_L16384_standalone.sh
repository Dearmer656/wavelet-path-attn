#!/bin/bash
#SBATCH --job-name=alibimed_hp_L16384
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_alibi_medium_s42_L16384_standalone.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:2
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Standalone L16384 HotpotQA-Long eval for ALiBi medium, run in parallel on separate GPUs
# while eval_alibi_medium_mix_s42_hotpot_alllen.sh (578004) is still working through
# L12288 sequentially -- same checkpoint-15000 as that script (not the true final
# checkpoint-15900) for consistency with the other 5 already-landed lengths in that same
# sweep. bf16 fallback included since L16384 is the longest/most OOM-prone length.

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
echo "=== ALiBi medium (finetuned) s42 HotpotQA-Long L${BSIZE} (standalone) ==="
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
  --block_size "${BSIZE}" \
  --per_device_eval_batch_size 1 \
  --output_dir "${OUTPUT}" --overwrite_output_dir \
  --logging_dir "${OUTPUT}/log" \
  --ddp_timeout 21600 --seed 42 --load_best_model_at_end False
python3 -c "import json; d=json.load(open('${OUTPUT}/eval_results.json')); print(f'ALiBi medium (finetuned) s42 L${BSIZE}: F1={d[\"eval_f1\"]:.4f} eval_loss={d[\"eval_loss\"]:.4f}')"

echo "=== Done: ALiBi medium (finetuned) s42 HotpotQA-Long L16384 (standalone) ==="

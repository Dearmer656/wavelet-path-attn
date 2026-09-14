#!/bin/bash
#SBATCH --job-name=rotary_yarn_medium_L2048_noaf
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/log_file/train/%j_rotary_yarn_medium_L2048_noattnfactor.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:p6000:3
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Isolates whether YaRN's attention_factor (mscale, a uniform q/k temperature
# scale ~1.1386 at factor=4) is what's suppressing HotpotQA F1, vs the frequency
# rescale itself. YaRN's frequency table (idx>=16 ratio to plain: 0.25, i.e. a
# plain 1/factor interpolation for the low-frequency/long-range dims, near-1.0
# for high-frequency/local dims -- an NTK-by-parts ramp) isn't wildly different
# from plain NTK's uniform rescale (verified by direct comparison), yet plain
# NTK gave F1=0.5875 at this same L2048 while YaRN gave only 0.0822. Runs with
# yarn_disable_attention_factor=true (new diagnostic-only cfg flag) to see if
# removing just the attention_factor scaling recovers most of the gap.

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
OUTPUT="${BASE}/hotpot_long/results/rotary_yarn_medium_s42_ckpt15000_noattnfactor/L${BSIZE}"
mkdir -p "${OUTPUT}/log"

CFG_PATH="${OUTPUT}/supply_model.cfg"
cat > "${CFG_PATH}" <<'CFG'
yarn_disable_attention_factor=true
CFG

echo "=== YaRN medium L2048, attention_factor DISABLED (diagnostic) ==="
MASTER_PORT=$(( 31500 + SLURM_JOB_ID % 1000 ))
python -m torch.distributed.run --nproc_per_node=3 --master_port=${MASTER_PORT} ./run_clm.py \
  --model_type gpt2 --tokenizer_name gpt2 \
  --model_name_or_path "${CKPT}" \
  --attn_implementation eager \
  --pe_method rotary \
  --use_yarn True \
  --yarn_factor 4 \
  --yarn_original_max_position_embeddings 512 \
  --dataset_name hotpot_qa --dataset_config_name distractor \
  --hotpot_long_jsonl "${JSONL}" \
  --hotpot_long_lengths "${BSIZE}" \
  --do_eval \
  --block_size "${BSIZE}" \
  --per_device_eval_batch_size 4 \
  --output_dir "${OUTPUT}" --overwrite_output_dir \
  --logging_dir "${OUTPUT}/log" \
  --seed 42 --load_best_model_at_end False \
  --cfg_path "${CFG_PATH}"
python3 -c "import json; d=json.load(open('${OUTPUT}/eval_results.json')); print(f'YaRN medium L${BSIZE} (attn_factor DISABLED): F1={d[\"eval_f1\"]:.4f} EM={d[\"eval_em\"]:.4f} eval_loss={d[\"eval_loss\"]:.4f}')"

echo "=== Done ==="

#!/bin/bash
#SBATCH --job-name=rotyarn_L2048_fixed
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/log_file/train/%j_rotary_yarn_medium_s42_L2048_fixed.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:2
#SBATCH --time=25:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Rerun of the zero-shot YaRN L2048 eval after fixing the checkpoint-load bug
# (rotary_embedding_torch's `freqs` is an nn.Parameter, part of state_dict --
# from_pretrained's state_dict load was silently overwriting the __init__-time
# YaRN rescale with the checkpoint's original plain-RoPE frequencies every time,
# for every YaRN run to date. run_clm.py now reapplies YaRN's frequency rescale
# right after from_pretrained returns, verified directly (freqs now differ from a
# plain, non-YaRN load of the same checkpoint). Old (buggy) result for comparison:
# rotary_yarn_medium_s42_ckpt15000/L2048 F1=0.0822 (statistically identical to
# plain RoPE's 0.0805, because YaRN was never actually applied).

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
YARN_FACTOR=$(python3 -c "print(${BSIZE}/512)")
OUTPUT="${BASE}/hotpot_long/results/rotary_yarn_medium_s42_ckpt15000_fixed/L${BSIZE}"
mkdir -p "${OUTPUT}/log"
echo "=== Rotary+YaRN medium (zero-shot, BUGFIXED) s42 HotpotQA-Long L${BSIZE} (yarn_factor=${YARN_FACTOR}) ==="
MASTER_PORT=$(( 26500 + SLURM_JOB_ID % 1000 ))
python -m torch.distributed.run --nproc_per_node=2 --master_port=${MASTER_PORT} ./run_clm.py \
  --model_type gpt2 --tokenizer_name gpt2 \
  --model_name_or_path "${CKPT}" \
  --attn_implementation eager \
  --pe_method rotary \
  --use_yarn True \
  --yarn_factor "${YARN_FACTOR}" \
  --yarn_original_max_position_embeddings 512 \
  --dataset_name hotpot_qa --dataset_config_name distractor \
  --hotpot_long_jsonl "${JSONL}" \
  --hotpot_long_lengths "${BSIZE}" \
  --do_eval \
  --block_size "${BSIZE}" \
  --per_device_eval_batch_size 1 \
  --output_dir "${OUTPUT}" --overwrite_output_dir \
  --logging_dir "${OUTPUT}/log" \
  --seed 42 --load_best_model_at_end False
python3 -c "import json; d=json.load(open('${OUTPUT}/eval_results.json')); print(f'Rotary+YaRN medium (zero-shot, FIXED) s42 L${BSIZE}: F1={d[\"eval_f1\"]:.4f} eval_loss={d[\"eval_loss\"]:.4f}')"

echo "=== Done: Rotary+YaRN medium (zero-shot, FIXED) s42 L2048 ==="

#!/bin/bash
#SBATCH --job-name=verify_ref_L512
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/log_file/train/%j_verify_reference_L512.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=0:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Reference (single-GPU, standard non-head-parallel eager ALiBi) eval on the SAME 200-sample
# subset as _verify_head_parallel_L512.sh, for an apples-to-apples correctness comparison
# (the full-dataset F1=0.8102 is over ~7363 examples, not directly comparable to a 200-case
# subset).

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
JSONL="${BASE}/hotpot_long/data/hotpot_long_dev.jsonl"
OUTPUT="${BASE}/hotpot_long/results/_head_parallel_verify/L512_reference"
mkdir -p "${OUTPUT}/log"
cd "${BASE}"
MASTER_PORT=$(( 15600 + SLURM_JOB_ID % 10000 ))

python -m torch.distributed.run --nproc_per_node=1 --master_port=${MASTER_PORT} ./run_clm.py \
  --model_type gpt2 --tokenizer_name gpt2 \
  --model_name_or_path "${CKPT}" \
  --attn_implementation eager \
  --pe_method alibi \
  --bf16 True \
  --dataset_name hotpot_qa --dataset_config_name distractor \
  --hotpot_long_jsonl "${JSONL}" \
  --hotpot_long_lengths 512 \
  --do_eval \
  --max_eval_samples 200 \
  --block_size 512 \
  --per_device_eval_batch_size 1 \
  --output_dir "${OUTPUT}" --overwrite_output_dir \
  --logging_dir "${OUTPUT}/log" \
  --ddp_timeout 21600 --seed 42 --load_best_model_at_end False

python3 -c "import json; d=json.load(open('${OUTPUT}/eval_results.json')); print(f'reference (single-GPU, non-head-parallel) L512 (200 samples): F1={d[\"eval_f1\"]:.4f} eval_loss={d[\"eval_loss\"]:.4f}')"

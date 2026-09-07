#!/bin/bash
#SBATCH --job-name=verify_headpar_L512
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/log_file/train/%j_verify_head_parallel_L512.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:2
#SBATCH --time=0:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Sanity check: head-parallel eager-ALiBi attention on 2 ranks at L512, where the known
# correct answer (F1=0.8102, eval_loss=1.0542) already exists from the standard
# single-process eager sweep -- confirms the head-split + all_gather implementation is
# numerically correct before trusting it on the expensive L16384 run.

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export ALIBI_HEAD_PARALLEL_L=512
export ALIBI_HEAD_PARALLEL_MAX_SAMPLES=200
export ALIBI_HEAD_PARALLEL_OUTPUT="/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/results/_head_parallel_verify/L512"
mkdir -p "${ALIBI_HEAD_PARALLEL_OUTPUT}/log"

BASE=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
cd "${BASE}"
MASTER_PORT=$(( 15500 + SLURM_JOB_ID % 10000 ))

python -m torch.distributed.run --nproc_per_node=2 --master_port=${MASTER_PORT} \
  _head_parallel_alibi_eval.py

python3 -c "import json; d=json.load(open('${ALIBI_HEAD_PARALLEL_OUTPUT}/eval_results.json')); print(f'head-parallel L512 (200 samples): F1={d[\"eval_f1\"]:.4f} eval_loss={d[\"eval_loss\"]:.4f}')"

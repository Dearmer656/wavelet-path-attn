#!/bin/bash
#SBATCH --job-name=PAT310_foxqwab_verify
#SBATCH --output=/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_pat310_pathfox_qwab/logs/%j_verify.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:1
#SBATCH --exclude=elm72,elm26
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
set -uxo pipefail
set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID:-manual}
cd /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
python /project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_pat310_pathfox_qwab/verify_gate_qwab.py

#!/bin/bash
#SBATCH --job-name=pat310_foxdecay
#SBATCH --output=/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat310_fox/%j_foxdecay.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:p6000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=100:00:00
set -euo pipefail
set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
cd /project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling
python probe_fox_forget_decay.py

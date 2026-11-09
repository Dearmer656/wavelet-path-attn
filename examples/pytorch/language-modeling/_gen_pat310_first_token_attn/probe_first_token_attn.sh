#!/bin/bash
#SBATCH --job-name=PAT310_firsttok_attn
#SBATCH --output=/cl/work5/hongyu-s/tmp_pat310/first_token_attn/logs/%j.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:p6000:1
#SBATCH --exclude=elm72,elm26
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
set -eo pipefail
source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets WANDB_DISABLED=true WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
python probe_first_token_attn.py --n 100 --lens 512,2048,4096

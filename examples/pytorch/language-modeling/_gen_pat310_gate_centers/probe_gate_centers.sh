#!/bin/bash
#SBATCH --job-name=PAT310_gate_centers
#SBATCH --output=/cl/work5/hongyu-s/tmp_pat310/gate_probe/logs/%j_centers.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:1
#SBATCH --exclude=elm72,elm26
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
set -eo pipefail
source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac WANDB_DISABLED=true
cd /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
python probe_gate_centers.py

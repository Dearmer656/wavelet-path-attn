#!/bin/bash
#SBATCH --job-name=cmp_K4_gain
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_compare_K4_s42_gain.txt
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:3090:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00

# CPU-only analysis (safetensors + matplotlib, no GPU code), but the lang_*
# CPU nodes (ahcclcsa01-04) are all down for maintenance right now, so
# borrowing a single 3090 slot at user's explicit direction instead of
# waiting out the maintenance window.

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u
  source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh
  conda activate latest_transformers
  set -u
fi

cd /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
python3 compare_pat234_K4_s42_withnull_vs_independent_gain.py

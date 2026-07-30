#!/bin/bash
#SBATCH --job-name=cmp_K4_gain
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_compare_K4_s42_gain.txt
#SBATCH --partition=lang_short
#SBATCH --account=lang
#SBATCH --nodelist=ahcclcsa01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u
  source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh
  conda activate latest_transformers
  set -u
fi

cd /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
python3 compare_pat234_K4_s42_withnull_vs_independent_gain.py

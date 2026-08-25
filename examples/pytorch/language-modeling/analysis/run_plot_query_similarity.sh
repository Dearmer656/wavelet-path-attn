#!/bin/bash
#SBATCH --job-name=plot_query_similarity
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/analysis/logs/%j_plot_query_similarity.txt
#SBATCH --partition=gpu_short
#SBATCH --nodelist=elm52
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=0:10:00
set -euxo pipefail
if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u; fi
cd /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
python analysis/plot_query_similarity_heatmap.py

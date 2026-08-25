#!/bin/bash
#SBATCH --job-name=query_similarity_nope_vs_path
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/analysis/logs/%j_query_similarity.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --nodelist=elm52
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
set -euxo pipefail
if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u; fi
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
cd /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
python analysis/query_similarity_nope_vs_path.py

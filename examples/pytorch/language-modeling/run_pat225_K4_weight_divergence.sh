#!/bin/bash
#SBATCH --job-name=pat225_weightdiv
#SBATCH --output=/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/logs/%j_weight_divergence.txt
#SBATCH --partition=lang_short
#SBATCH --account=lang
#SBATCH --nodelist=ahcclcsa01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00

set -euxo pipefail
if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi
cd /project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling
python analyze_pat225_K4_weight_divergence.py

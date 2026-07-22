#!/bin/bash
#SBATCH --job-name=pat234_dose
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat234_scale_card/_dose_%j.txt
#SBATCH --partition=lang_short
#SBATCH --account=lang
#SBATCH --nodelist=ahcclcsa01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=0:20:00
set -euo pipefail
if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi
cd /project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling
python ./aggregate_pat234_dose_response.py "${1:-4096}"

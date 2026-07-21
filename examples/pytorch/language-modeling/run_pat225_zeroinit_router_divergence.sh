#!/bin/bash
#SBATCH --job-name=pat225_zeroinit_router_div
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/logs/%j_zeroinit_router_div.txt
#SBATCH --partition=lang_short
#SBATCH --account=lang
#SBATCH --nodelist=ahcclcsa01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00

set -euxo pipefail
if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi
cd /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
python analyze_pat225_zeroinit_router_divergence.py
echo "=== done ==="

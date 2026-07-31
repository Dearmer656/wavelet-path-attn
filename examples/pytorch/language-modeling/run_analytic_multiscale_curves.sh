#!/bin/bash
#SBATCH --job-name=analytic_curves
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_analytic_curves.txt
#SBATCH --partition=gpu_short
#SBATCH --nodelist=elm26
#SBATCH --gres=gpu:q6000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=0:20:00

# CPU-only (numpy+matplotlib, no torch/GPU code); lang_* CPU nodes are down
# for maintenance, so borrowing a single idle GPU slot just for a compute
# node, same workaround used earlier in this session.
set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u
  source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh
  conda activate latest_transformers
  set -u
fi

WORKDIR=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
cd "${WORKDIR}"

python3 plot_analytic_multiscale_curves.py --output_dir "${WORKDIR}/analysis_outputs/analytic_multiscale_curves"

echo "=== DONE analytic_multiscale_curves ==="

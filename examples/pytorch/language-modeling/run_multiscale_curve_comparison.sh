#!/bin/bash
#SBATCH --job-name=multiscale_curve
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_multiscale_curve.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u
  source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh
  conda activate latest_transformers
  set -u
fi

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true
export WANDB_MODE=disabled

WORKDIR=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
cd "${WORKDIR}"

echo "=== K1/K2/K4/K5/K6 multi-scale bias curve comparison ==="
python3 plot_multiscale_curve_comparison.py \
  --checkpoint_step 15000 \
  --eval_length 512 \
  --num_samples 64 \
  --batch_size 8 \
  --seed 42 \
  --device cuda \
  --dtype bfloat16 \
  --output_dir "${WORKDIR}/analysis_outputs/multiscale_curve_comparison"

echo "=== DONE multiscale_curve_comparison ==="

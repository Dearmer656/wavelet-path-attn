#!/bin/bash
#SBATCH --job-name=k1scales_vs_k4
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_k1_scales_vs_k4.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00

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

echo "=== K4 per-scale (me=8,16,20,24) vs matching standalone K1_meN amplitude comparison, L512 ==="
python3 analyze_k1_scales_vs_k4.py \
  --k4_run "${WORKDIR}/runs/pat234_scale_card/K4_me8_16_20_24_noC1_s42_sqrtnorm" \
  --eval_length 512 \
  --num_samples 128 \
  --batch_size 4 \
  --seed 42 \
  --device cuda \
  --dtype bfloat16 \
  --output_dir "${WORKDIR}/analysis_outputs/k1_scales_vs_k4"

echo "=== DONE k1_scales_vs_k4 ==="

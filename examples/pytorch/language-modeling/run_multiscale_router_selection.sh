#!/bin/bash
#SBATCH --job-name=PAT234_multiscale_router_sel
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/pat234_multiscale_router_sel/%j_multiscale_router_sel.txt
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

cd /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets

mkdir -p pat234_multiscale_router_sel

python analyze_multiscale_router_selection.py \
  --runs_root runs/pat234_scale_card \
  --run_dirs \
    K3_me14_16_17p1699_noC1_s42_independent_rms_center0_ricker_elm73_6000x4:K3 \
    K5_me14_15p1998_16_16p6439_17p1699_noC1_s42_independent_rms_center0_ricker_lang01_a100x4:K5 \
    K4_me8_16_20_24_noC1_s42_independent_rms_center0_ricker_elm43_a100x4:K4 \
  --eval_length 512 \
  --num_samples 32 \
  --micro_batch_size 8 \
  --seed 42 \
  --device cuda \
  --dtype float32

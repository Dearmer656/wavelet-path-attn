#!/usr/bin/env bash
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u; fi

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export WANDB_DISABLED=true

cd /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
python analysis/spectral/spectral_pa_vs_qwab.py

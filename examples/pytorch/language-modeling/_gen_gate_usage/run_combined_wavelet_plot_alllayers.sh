#!/bin/bash
#SBATCH --job-name=plot_wavelet_alllayers_ckpt15000
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/analysis/k3_signed_rms/logs/%j_plot_wavelet_alllayers.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:1 --nodelist=elm63
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
set -euxo pipefail
if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u; fi
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
cd /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
python3 analysis/k3_signed_rms/plot_combined_wavelet_pattern_alllayers.py

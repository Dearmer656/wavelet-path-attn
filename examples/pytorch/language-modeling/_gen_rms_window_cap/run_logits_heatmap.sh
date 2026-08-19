#!/bin/bash
#SBATCH --job-name=logits_heatmap_pa_vs_qwab
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_rms_window_cap/logs/%j_logits_heatmap.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
set -euxo pipefail
if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u; fi
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
cd /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
python3 /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_rms_window_cap/plot_logits_heatmap_pa_vs_qwab.py

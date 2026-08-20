#!/bin/bash
#SBATCH --job-name=logit_heatmap_512bins
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_postfix_4way_s42/logs/%j_logit_heatmap_512bins.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
set -euxo pipefail
if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u; fi
cd /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_postfix_4way_s42
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python ./plot_logit_heatmap_512bins.py
echo "=== done ==="

#!/bin/bash
#SBATCH --job-name=dump_train_cases
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_rms_window_cap/logs/%j_dump_train_cases.txt
#SBATCH --partition=gpu_short
#SBATCH --nodelist=elm26
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
set -euxo pipefail
if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u; fi
cd /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_rms_window_cap
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_HUB_OFFLINE=1
export WANDB_DISABLED=true
export CUDA_VISIBLE_DEVICES=""
python ./dump_train_cases.py
echo "=== done ==="

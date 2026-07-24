#!/bin/bash
#SBATCH --job-name=clamp_normoff
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat234_scale_card/_clampnoff_%j.txt
#SBATCH --partition=lang_long
#SBATCH --account=lang
#SBATCH --nodelist=ahcclcsa01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
set -euxo pipefail
if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u; fi
cd /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac; export HF_HUB_OFFLINE=1; export WANDB_DISABLED=true; export CUDA_VISIBLE_DEVICES=""
python ./probe_clamp_norm_off.py
echo "=== done ==="

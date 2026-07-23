#! /bin/bash
#SBATCH --job-name=bb_imprint
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat234_scale_card/_bbimprint_%j.txt
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=1:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
set -euxo pipefail
if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi
cd /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export WANDB_DISABLED=true; export WANDB_MODE=disabled; export HF_HUB_OFFLINE=1
python ./probe_backbone_imprint.py
echo "=== done ==="

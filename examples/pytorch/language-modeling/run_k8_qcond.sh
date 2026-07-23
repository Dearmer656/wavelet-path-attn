#! /bin/bash
#SBATCH --job-name=k8_qcond
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat234_scale_card/_k8qcond_%j.txt
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=0:30:00
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
python ./probe_k8_query_conditioning.py "${1:-hotpot}" "${2:-2048}"
echo "=== done ==="

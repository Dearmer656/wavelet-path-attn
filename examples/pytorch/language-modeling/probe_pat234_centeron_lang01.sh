#! /bin/bash
#SBATCH --job-name=PAT234_visprobe2500
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat234_scale_card/_probe2500_%j.txt
#SBATCH --partition=lang_gpu_long
#SBATCH --account=lang
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# PAT-234: as-trained softmax-visible amplitude (M_eff) probe on the center-ON
# K=1 sweep checkpoints. Runs on lang01 (separate QOS cap) so it does not touch
# the is-nlp 12-GPU budget held by the 5 training jobs.
set -euxo pipefail
if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi
WORKDIR=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
cd "${WORKDIR}"
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
STEP="${1:-2500}"
echo "================= PAT-234 visible-amplitude probe, step ${STEP} ================="
python ./probe_pat234_centeron_visible.py "${STEP}"
echo "=== probe done ==="

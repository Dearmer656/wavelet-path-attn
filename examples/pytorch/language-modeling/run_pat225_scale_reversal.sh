#! /bin/bash
#SBATCH --job-name=PAT225_scale_reversal
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/logs/%j_pat225_scale_reversal.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# PAT-225 follow-up: scan per-scale router pi across all 12 layers and all
# saved checkpoints of S4_s42 (orig, best), S4_s42_rerun (worse), and
# S4_s42_zeroinit (in-progress validation) to test whether the router's
# dominant scale identity reverses (A->B->A) rather than just monotonically
# sharpening/flattening.

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi

WORKDIR=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
cd "${WORKDIR}"

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets

python analyze_pat225_scale_reversal.py

echo "=== PAT-225 scale reversal scan done ==="

#! /bin/bash
#SBATCH --job-name=PAT225_k1_response
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/logs/%j_k1_scale_response.txt
#SBATCH --partition=lang_gpu_long
#SBATCH --account=lang
#SBATCH --nodelist=lang01
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

set -euxo pipefail
if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi
WORKDIR=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
cd "${WORKDIR}"
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
python analyze_pat225_k1_scale_response.py
echo "=== PAT-225 K=1 scale-response probe done ==="

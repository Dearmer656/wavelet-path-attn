#! /bin/bash
#SBATCH --job-name=PAT225_K4scalecmp
#SBATCH --output=/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat225_scale_card/analysis_logs/%j_scale_seedcompare.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:q6000:1
#SBATCH --nodelist=elm26
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets

cd /project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling
python analyze_pat225_K4_scale_seedcompare.py

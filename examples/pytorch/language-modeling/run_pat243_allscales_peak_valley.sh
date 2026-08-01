#!/bin/bash
#SBATCH --job-name=PAT243_allscales_peak_valley
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/pat243_allscales_peak_valley/%j_pat243_allscales_peak_valley.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u
  source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh
  conda activate latest_transformers
  set -u
fi

cd /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets

mkdir -p pat243_allscales_peak_valley

python analyze_k1_allscales_peak_valley.py \
  --runs_root runs/pat234_scale_card \
  --output_dir pat243_allscales_peak_valley \
  --eval_lengths 512 4096 \
  --num_samples 32 \
  --seed 42 \
  --device cuda \
  --dtype float32 \
  --micro_batch_size_512 8 \
  --micro_batch_size_4096 2

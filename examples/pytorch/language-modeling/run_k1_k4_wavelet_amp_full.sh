#!/bin/bash
#SBATCH --job-name=k1k4_amp_full
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_k1k4_amp_full.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u
  source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh
  conda activate latest_transformers
  set -u
fi

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true
export WANDB_MODE=disabled

WORKDIR=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
cd "${WORKDIR}"

echo "=== full K1 vs K4 wavelet amp analysis, L512 only (main result, must not be blocked by longer lengths) ==="
python3 analyze_k1_k4_wavelet_amp.py \
  --k1_run "${WORKDIR}/runs/pat234_scale_card/K1_me16_noC1_s42" \
  --k4_run "${WORKDIR}/runs/pat234_scale_card/K4_me8_16_20_24_noC1_s42_sqrtnorm" \
  --eval_lengths 512 \
  --num_samples 128 \
  --batch_size 4 \
  --seed 42 \
  --device cuda \
  --dtype bfloat16 \
  --output_dir "${WORKDIR}/analysis_outputs/k1_k4_wavelet_amp" \
  --skip_sanity_check

echo "=== DONE L512 full analysis ==="

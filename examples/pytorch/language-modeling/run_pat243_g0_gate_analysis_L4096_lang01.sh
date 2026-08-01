#!/bin/bash
#SBATCH --job-name=PAT243_g0_gate_heatmap_L4096
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/pat243_g0_gate_analysis_L4096/%j_pat243_g0_gate_heatmap_L4096_lang01.txt
#SBATCH --partition=lang_gpu_long
#SBATCH --account=lang
#SBATCH --qos=qos_lang
#SBATCH --nodelist=lang01
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00

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

mkdir -p pat243_g0_gate_analysis_L4096

python analyze_k1_g0_gate_by_checkpoint.py \
  --run_dir runs/pat234_scale_card/K1_me16_noC1_s42_center0_ricker_norouterrms_lang01_a100x2 \
  --output_dir pat243_g0_gate_analysis_L4096 \
  --eval_length 4096 \
  --num_samples 32 \
  --seed 42 \
  --device cuda \
  --dtype float32 \
  --micro_batch_size 2

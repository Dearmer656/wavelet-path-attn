#!/bin/bash
#SBATCH --job-name=xsumfill_L256_PaTH_s43
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/log_file/train/%j_xsumfill_L256_PaTH_s43.txt
#SBATCH --partition=gpu_long
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:3090:1
#SBATCH --exclude=elm72,elm26
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
set -uxo pipefail
bash /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_xsum_basis/eval_filter_xsum_bias_fixed_1gpu.sh /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat310_L256train/PaTH_s43 off 42 L256tr_PaTH_s43

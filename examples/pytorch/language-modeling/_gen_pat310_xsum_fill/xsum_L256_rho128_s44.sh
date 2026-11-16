#!/bin/bash
#SBATCH --job-name=xsumfill_L256_rho128_s44
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/log_file/train/%j_xsumfill_L256_rho128_s44.txt
#SBATCH --partition=gpu_long
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:3090:1
#SBATCH --exclude=elm72,elm26
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
set -uxo pipefail
bash /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_xsum_basis/eval_filter_xsum_bias_fixed_1gpu.sh /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat244_dual_temp/K1_L256_me14_rho128_ricker_s44 logit_bias_ctxscale_shift_v0 42 L256tr_rho128_s44

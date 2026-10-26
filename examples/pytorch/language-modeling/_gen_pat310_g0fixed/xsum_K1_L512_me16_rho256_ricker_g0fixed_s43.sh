#!/bin/bash
#SBATCH --job-name=PAT310_xsum_g0fixed_s43
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/log_file/train/%j_PAT310_xsum_g0fixed_s43.txt
#SBATCH --partition=gpu_long
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:a6000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
set -uxo pipefail
bash /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_xsum_basis/eval_filter_xsum_bias_fixed_1gpu.sh /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat310_g0fixed/K1_L512_me16_rho256_ricker_g0fixed_s43 logit_bias_ctxscale_shift_v0 42 K1_L512_me16_rho256_ricker_g0fixed_s43

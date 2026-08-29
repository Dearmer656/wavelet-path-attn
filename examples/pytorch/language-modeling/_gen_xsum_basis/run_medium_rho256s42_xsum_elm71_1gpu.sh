#!/bin/bash
#SBATCH --job-name=xsum_medK1rho256_s42
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/head_wise_scale_selection_vs_layer_wise/layer_wise/sigmoid_exp/eval_log/xsum_medK1rho256_s42_%j.txt
#SBATCH --partition=gpu_long
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:a6000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
set -uxo pipefail
bash /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_xsum_basis/eval_filter_xsum_bias_fixed_1gpu.sh \
  /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/mix_medium_K1_rho256_ricker_s42 \
  logit_bias_ctxscale_shift_v0 42 medK1rho256_s42

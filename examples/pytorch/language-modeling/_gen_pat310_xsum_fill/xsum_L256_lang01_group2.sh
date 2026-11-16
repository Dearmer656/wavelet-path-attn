#!/bin/bash
#SBATCH --job-name=xsumfill_L256_g2
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/log_file/train/%j_xsumfill_L256_g2.txt
#SBATCH --partition=lang_gpu_long
#SBATCH --account=lang
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=100:00:00
# 4 independent single-GPU Filtered XSum evals in parallel (one per GPU); SLURM_JOB_ID is overridden per eval only to give each a distinct torchrun port and run tag.
set -uxo pipefail
BASE=$SLURM_JOB_ID
IFS=, read -ra DEV <<< "$CUDA_VISIBLE_DEVICES"
echo "allocated GPUs: ${DEV[*]}"
CUDA_VISIBLE_DEVICES=${DEV[0]} SLURM_JOB_ID=$((BASE*10+0)) bash /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_xsum_basis/eval_filter_xsum_bias_fixed_1gpu.sh /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat244_dual_temp/K1_L256_me14_rho128_ricker_s43 logit_bias_ctxscale_shift_v0 42 L256tr_rho128_s43 > /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/log_file/train/${BASE}_xsumfill_L256tr_rho128_s43.txt 2>&1 &
CUDA_VISIBLE_DEVICES=${DEV[1]} SLURM_JOB_ID=$((BASE*10+1)) bash /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_xsum_basis/eval_filter_xsum_bias_fixed_1gpu.sh /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat244_dual_temp/K1_L256_me14_rho128_ricker_s44 logit_bias_ctxscale_shift_v0 42 L256tr_rho128_s44 > /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/log_file/train/${BASE}_xsumfill_L256tr_rho128_s44.txt 2>&1 &
CUDA_VISIBLE_DEVICES=${DEV[2]} SLURM_JOB_ID=$((BASE*10+2)) bash /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_xsum_basis/eval_filter_xsum_bias_fixed_1gpu.sh /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat244_dual_temp/K1_L256_me16_rho256_ricker_s42 logit_bias_ctxscale_shift_v0 42 L256tr_rho256_s42 > /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/log_file/train/${BASE}_xsumfill_L256tr_rho256_s42.txt 2>&1 &
CUDA_VISIBLE_DEVICES=${DEV[3]} SLURM_JOB_ID=$((BASE*10+3)) bash /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_xsum_basis/eval_filter_xsum_bias_fixed_1gpu.sh /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat244_dual_temp/K1_L256_me16_rho256_seed43 logit_bias_ctxscale_shift_v0 42 L256tr_rho256_s43 > /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/log_file/train/${BASE}_xsumfill_L256tr_rho256_s43.txt 2>&1 &
wait

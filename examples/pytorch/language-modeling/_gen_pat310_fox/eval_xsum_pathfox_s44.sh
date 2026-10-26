#!/bin/bash
#SBATCH --job-name=PAT310_xsumFoX_s44
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/log_file/train/%j_PAT310_xsumFoX_s44.txt
#SBATCH --partition=gpu_long
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:a6000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
set -uxo pipefail
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID:-manual}
# PAT-310 W5 PaTH-FoX (triton) small XSum, paper protocol (same helper as Table 4 basis runs)
bash /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_pat310_fox/eval_filter_xsum_fox_triton_1gpu.sh /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat310_fox/small_pathfox_s44 off 44 small_pathfox_s44

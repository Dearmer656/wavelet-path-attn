#!/bin/bash
#SBATCH --job-name=xsum_pathfox_b5_s42
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat310_medium_pathfox/xsum_eval_log/xsum_pathfox_b5_s42_%j.txt
#SBATCH --partition=gpu_long
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:a6000:1
#SBATCH --exclude=elm72
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
set -uxo pipefail
bash /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_pat310_medium_pathfox/eval_filter_xsum_pathfox_1gpu.sh /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat310_medium_pathfox/pathfox_b5_s42 db1 42 pathfox_b5_s42

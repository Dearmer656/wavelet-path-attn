#!/bin/bash
#SBATCH --job-name=pat310_transport
#SBATCH --output=/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat310_c2/transport_probe/logs/%j.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=100:00:00
set -euo pipefail
set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
cd /project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling
python probe_path_transport_far_near.py "$@"

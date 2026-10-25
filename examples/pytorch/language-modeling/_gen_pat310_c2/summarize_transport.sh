#!/bin/bash
#SBATCH --job-name=tp_sum
#SBATCH --output=/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat310_c2/transport_probe/logs/sum_%j.txt
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate base; set -u
python /project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/summarize_path_transport.py "$@"

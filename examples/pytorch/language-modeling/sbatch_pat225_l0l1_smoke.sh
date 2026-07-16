#!/bin/bash
#SBATCH --job-name=pat225_smoke
#SBATCH --output=/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/log_file/train/%j_pat225_smoke.txt
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:a6000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00

# PAT-225 L0/L1 gate: grid regression (A4 bit-exact reuse guard) + S=1/2/4 smoke.
# Branch: hongyusaatitech/pat-225-sensei-ablation-is-qwab-genuinely-multi-scale-scale
# (both transformers and flash-linear-attention repos)

set -euxo pipefail

export PYTHONPATH="/project/nlp-work5/hongyu-s/flash-linear-attention:/project/nlp-work5/hongyu-s/transformers/src:${PYTHONPATH:-}"
PYTHON="/cl/work5/hongyu-s/conda/envs/latest_transformers/bin/python"

cd /project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling
${PYTHON} pat225_smoke_and_regression.py --parts abc
echo "=== PAT-225 L0/L1 gate PASS ==="

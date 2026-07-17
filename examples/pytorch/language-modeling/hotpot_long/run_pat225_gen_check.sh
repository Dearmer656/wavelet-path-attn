#!/bin/bash
#SBATCH --job-name=pat225_gen
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_pat225_gen_check.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# PAT-225 generation-based validation (open-ended decoding, not teacher-forced).
# Checks whether the TF-protocol headline ordering (S4 > S8 > PA at L4096)
# survives under model.generate. Subsampled to 1000 examples per length.
# Usage: sbatch run_pat225_gen_check.sh <CHECKPOINT> <NAME> <CFG_PATH>

set -euxo pipefail

CHECKPOINT="${1:?CHECKPOINT required}"
NAME="${2:?NAME required}"
CFG_PATH="${3:?CFG_PATH required}"

LM=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
HL="${LM}/hotpot_long"
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
PYTHON="/cl/work5/hongyu-s/conda/envs/latest_transformers/bin/python"

OUT="${HL}/results_gen_check/${NAME}"
mkdir -p "${OUT}"
cd "${LM}"

${PYTHON} "${HL}/eval_hotpot_long.py" \
    --model-path "${CHECKPOINT}" \
    --hotpot-long-jsonl "${HL}/data/hotpot_long_dev_uniform.jsonl" \
    --tokenizer gpt2 \
    --output-dir "${OUT}" \
    --batch-size 4 \
    --target-lengths 512 4096 \
    --max-new-tokens 32 \
    --max-examples 1000 \
    --device auto \
    --n-boot 2000 \
    --cfg-path "${CFG_PATH}"

echo "=== gen-check ${NAME} done ==="
cat "${OUT}/summary.json" 2>/dev/null || true

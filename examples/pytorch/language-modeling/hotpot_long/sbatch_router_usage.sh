#!/bin/bash
#SBATCH --job-name=router_usage
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_router_usage.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:1
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# Generic per-(model,seed) router-usage dump. Submit with
# --export=ALL,CKPT=...,MODEL_TAG=...,SEED=...,JOB_TAG=...,SEQ_LENS=...
# SEQ_LENS defaults to 512,4096 if not exported (small model fits this in
# fp32 on a 48GB card; medium needs SEQ_LENS=512,2048 -- L=4096 OOMs even a
# 48GB card purely from the model's own forward computation).

_slack() {
    python3 /project/nlp-work5/hongyu-s/gate1/scripts/notify_slack.py \
        --exit-code "$1" \
        --job-id "${SLURM_JOB_ID}" \
        --node "${SLURMD_NODENAME}" \
        --issue "PAT-253" \
        --gpu "6000x1" \
        --summary "K3 router usage dump: ${JOB_TAG}"
}
trap '_slack $?' EXIT

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export WANDB_DISABLED=true

cd /project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long

python dump_router_usage.py \
  --checkpoint "${CKPT}" \
  --model_tag "${MODEL_TAG}" \
  --seed "${SEED}" \
  --seq_lens "${SEQ_LENS:-512,4096}" \
  --n_case 50 \
  --jsonl data/hotpot_long_dev.jsonl \
  --out_csv "analysis_outputs/router_usage/${JOB_TAG}.csv"

echo "=== DONE: router usage ${JOB_TAG} ==="

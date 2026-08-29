#!/bin/bash
#SBATCH --job-name=router_usage_hp_mn
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_router_usage_hp_mn.txt
#SBATCH --partition=gpu_long
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:6000:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
# Cross-node 2xGPU head-parallel router-usage dump. First multi-node torchrun
# job in this project -- no single-node fallback if NCCL/rendezvous across
# nodes fails; that failure mode is expected to look like a hang or a
# connection-refused error in the log, not a python traceback.
#
# H_local=8 (16 heads / 2 ranks) vs. the proven 8xA100 config's H_local=2, so
# this targets L8192 (data/hotpot_long_dev_uniform_8192only.jsonl), not the
# original L16384 target -- memory-budget estimate only, unverified until
# this smoke test runs.
#
# Submit with --export=ALL,CKPT=...,MODEL_TAG=medium,SEED=...,JOB_TAG=...,N_CASE=...,SEQ_LEN=8192
# and --nodelist=elm72,elm73 (or whichever 2 same-availability nodes are free).

_slack() {
    python3 /project/nlp-work5/hongyu-s/gate1/scripts/notify_slack.py \
        --exit-code "$1" \
        --job-id "${SLURM_JOB_ID}" \
        --node "${SLURM_JOB_NODELIST}" \
        --issue "PAT-253" \
        --gpu "6000x2 (2 nodes, first multi-node job)" \
        --summary "K3 router usage dump @ L${SEQ_LEN:-8192} cross-node headparallel: ${JOB_TAG}"
}
trap '_slack $?' EXIT

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export WANDB_DISABLED=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO

cd /project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long

MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} NODELIST=${SLURM_JOB_NODELIST}"

SEQ_LEN="${SEQ_LEN:-8192}"

srun torchrun \
  --nnodes=2 \
  --nproc_per_node=1 \
  --rdzv_id="${SLURM_JOB_ID}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  dump_router_usage_headparallel.py \
  --checkpoint "${CKPT}" \
  --model_tag "${MODEL_TAG}" \
  --seed "${SEED}" \
  --seq_lens "${SEQ_LEN}" \
  --n_case "${N_CASE:-50}" \
  --jsonl "data/hotpot_long_dev_uniform_${SEQ_LEN}only.jsonl" \
  --out_csv "analysis_outputs/router_usage/${JOB_TAG}.csv" \
  --dtype "${DTYPE:-fp32}"

echo "=== DONE: router usage cross-node headparallel ${JOB_TAG} ==="

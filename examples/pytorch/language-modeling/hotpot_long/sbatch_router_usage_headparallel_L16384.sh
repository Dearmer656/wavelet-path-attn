#!/bin/bash
#SBATCH --job-name=router_usage_hp16384
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_router_usage_hp16384.txt
#SBATCH --partition=lang_gpu_long
#SBATCH --account=lang
#SBATCH --gres=gpu:a100:8
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# Medium K3 router-usage dump at L16384 via head-parallel (8xA100), mirroring
# the proven test_mix_medium_K3_ricker_128_256_384_s43_L16384_headparallel_8gpu.sh
# eval setup. Uses the uniform-pool dataset (front-placed data/hotpot_long_dev.jsonl
# tops out at L4096, no L16384 cases exist there -- user-approved deviation from
# the paper's front-placement citation convention, for this internal router-
# mechanism analysis only, not for paper tables).
# Submit with --export=ALL,CKPT=...,MODEL_TAG=medium,SEED=...,JOB_TAG=...,N_CASE=...

_slack() {
    python3 /project/nlp-work5/hongyu-s/gate1/scripts/notify_slack.py \
        --exit-code "$1" \
        --job-id "${SLURM_JOB_ID}" \
        --node "${SLURMD_NODENAME}" \
        --issue "PAT-253" \
        --gpu "a100x8" \
        --summary "K3 router usage dump @ L16384 headparallel: ${JOB_TAG}"
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

cd /project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long

MASTER_PORT=$((12000 + SLURM_JOB_ID % 10000))
torchrun --nproc_per_node=8 --master_port=${MASTER_PORT} dump_router_usage_headparallel.py \
  --checkpoint "${CKPT}" \
  --model_tag "${MODEL_TAG}" \
  --seed "${SEED}" \
  --seq_lens 16384 \
  --n_case "${N_CASE:-50}" \
  --jsonl data/hotpot_long_dev_uniform_16384_large_pool.jsonl \
  --out_csv "analysis_outputs/router_usage/${JOB_TAG}.csv"

echo "=== DONE: router usage headparallel ${JOB_TAG} ==="

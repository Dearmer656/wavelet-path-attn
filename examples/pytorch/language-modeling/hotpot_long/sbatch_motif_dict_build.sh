#!/bin/bash
#SBATCH --job-name=motif_dict_build
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_motif_dict_build.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:1
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# Generic per-length NMF motif dictionary builder for the PAT-194 motif-retention
# pipeline. Submit with --export=ALL,CKPT=...,JSONL=...,SEQ_LEN=...,CAPTURE_TOTAL=...,RUN_SUFFIX=...,JOB_TAG=...
# CAPTURE_TOTAL should be the literal string "--capture_total" or "" (empty).

_slack() {
    python3 /project/nlp-work5/hongyu-s/gate1/scripts/notify_slack.py \
        --exit-code "$1" \
        --job-id "${SLURM_JOB_ID}" \
        --node "${SLURMD_NODENAME}" \
        --issue "PAT-194" \
        --gpu "6000x1" \
        --summary "PAT-194 motif dict build: ${JOB_TAG}"
}
trap '_slack $?' EXIT

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export WANDB_DISABLED=true

OUT_ROOT=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/attn_nmf_comparison/motif_retention
mkdir -p "${OUT_ROOT}"
cd /project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long

python dump_path_nmf.py \
  --checkpoint "${CKPT}" \
  --jsonl "${JSONL}" \
  --out_root "${OUT_ROOT}" \
  --seq_len "${SEQ_LEN}" \
  --n_case 50 \
  --rank 16 \
  --pool_size 128 \
  --preprocessing salient \
  ${CAPTURE_TOTAL} \
  --run_suffix "${RUN_SUFFIX}"

echo "=== DONE: motif dict build ${JOB_TAG} ==="

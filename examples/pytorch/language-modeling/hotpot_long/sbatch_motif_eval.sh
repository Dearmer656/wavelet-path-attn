#!/bin/bash
#SBATCH --job-name=motif_eval
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_motif_eval.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:1
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# Generic motif-retention-gap evaluator (L2048 + L4096) for the PAT-194 pipeline.
# Submit with --export=ALL,MODEL=...,CKPT=...,TRAINDICT=...,ORACLE2048=...,ORACLE4096=...,JOB_TAG=...

_slack() {
    python3 /project/nlp-work5/hongyu-s/gate1/scripts/notify_slack.py \
        --exit-code "$1" \
        --job-id "${SLURM_JOB_ID}" \
        --node "${SLURMD_NODENAME}" \
        --issue "PAT-194" \
        --gpu "6000x1" \
        --summary "PAT-194 motif retention eval: ${JOB_TAG}"
}
trap '_slack $?' EXIT

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export WANDB_DISABLED=true

OUT_DIR=/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/analysis_outputs/motif_retention_gap
mkdir -p "${OUT_DIR}"
cd /project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long

python eval_motif_retention_gap.py \
  --model "${MODEL}" \
  --checkpoint "${CKPT}" \
  --traindict_dir "${TRAINDICT}" \
  --oracledict_dir_2048 "${ORACLE2048}" \
  --oracledict_dir_4096 "${ORACLE4096}" \
  --eval_jsonl data/motif_retention_pools/eval_pool.jsonl \
  --rank 16 --pool_size 128 --preprocessing salient \
  --out_dir "${OUT_DIR}"

echo "=== DONE: motif eval ${JOB_TAG} ==="

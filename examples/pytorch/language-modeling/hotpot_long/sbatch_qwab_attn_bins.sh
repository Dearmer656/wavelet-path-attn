#!/bin/bash
#SBATCH --job-name=qwab_attn_bins
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_qwab_attn_bins.txt
#SBATCH --partition=gpu_long
#SBATCH --nodelist=elm71
#SBATCH --gres=gpu:6000:1
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

set -euxo pipefail

_slack() {
    python3 /project/nlp-work5/hongyu-s/gate1/scripts/notify_slack.py \
        --exit-code "$1" \
        --job-id "${SLURM_JOB_ID}" \
        --node "${SLURMD_NODENAME}" \
        --issue "PAT-253" \
        --gpu "6000x1" \
        --summary "PAT-253 QWAB vs PA-only attn mass, qbin x lag-bin, K1 canonical ckpt, L512/L4096"
}
trap '_slack $?' EXIT

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export WANDB_DISABLED=true

cd /project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long

python dump_qwab_attn_bins.py \
  --checkpoint /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat244_dual_temp/K1_L512_me16_rho256_ricker_s42/checkpoint-15000 \
  --seq_lens 512,4096 \
  --n_case 50 \
  --jsonl data/hotpot_long_dev.jsonl \
  --n_qbin 10 \
  --out_dir analysis_outputs/qwab_attn_bins

echo "=== DONE ==="

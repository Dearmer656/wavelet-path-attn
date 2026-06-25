#!/bin/bash
#SBATCH --job-name=pat209_smoke
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_pat209_smoke.txt
#SBATCH --partition=gpu_short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:3090:1
#SBATCH --time=1:00:00

CKPT="${CKPT:-runs/rotary_mix_finetune/s42/checkpoint-15900}"
_slack() {
    python3 /project/nlp-work5/hongyu-s/gate1/scripts/notify_slack.py \
        --exit-code "$1" --job-id "${SLURM_JOB_ID}" --node "${SLURMD_NODENAME}" \
        --issue "PAT-209" --gpu "1x3090" --summary "PAT-209 step-2 smoke test" 2>/dev/null || true
}
trap '_slack $?' EXIT
set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
    set +u; . /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi
export PYTHONPATH=/cl/work5/hongyu-s/flash-linear-attention:/cl/work5/hongyu-s/transformers/src${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1

BASE=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
cd "${BASE}"
mkdir -p hotpot_long/logs
python hotpot_long/pat209_smoke_test.py --checkpoint "${CKPT}"
echo "=== Done: PAT-209 step-2 smoke ==="

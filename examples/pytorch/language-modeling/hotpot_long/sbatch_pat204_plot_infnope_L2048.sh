#!/bin/bash
#SBATCH --job-name=pat204_plot_infnope
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_pat204_plot_infnope.txt
#SBATCH --partition=gpu_short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:3090:1
#SBATCH --time=2:00:00
CASE_IDX="${CASE_IDX:-0}"
L="${L:-2048}"
_slack() { python3 /project/nlp-work5/hongyu-s/gate1/scripts/notify_slack.py --exit-code "$1" --job-id "${SLURM_JOB_ID}" --node "${SLURMD_NODENAME}" --issue "PAT-204" --gpu "1x3090" --summary "PAT-204 plot inference-NoPE8-11 L=${L} case=${CASE_IDX}" 2>/dev/null || true; }
trap '_slack $?' EXIT
set -euxo pipefail
if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then set +u; . /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u; fi
export PYTHONPATH=/cl/work5/hongyu-s/flash-linear-attention:/cl/work5/hongyu-s/transformers/src${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
BASE=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
cd "${BASE}/hotpot_long"
JSONL=${BASE}/hotpot_long/data/hotpot_long_dev_uniform.jsonl
OUT=${BASE}/hotpot_long/analysis_outputs/pat204_attn_maps
mkdir -p "${OUT}"
BASELINE=${BASE}/runs/rotary_mix_finetune/s42/checkpoint-15900
python plot_attention_maps.py --checkpoint "${BASELINE}" --jsonl "${JSONL}" --preset vertical --nope_layers "8-9-10-11" --length "${L}" --case_idx "${CASE_IDX}" --tag "infNoPE8to11" --out "${OUT}/case${CASE_IDX}_L${L}_infNoPE8to11.png"
echo "=== Done: inference-NoPE attn map ==="

#!/bin/bash
#SBATCH --job-name=hpqa_free_gen
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_hpqa_free_gen.txt
#SBATCH --partition=gpu_long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:4
#SBATCH --time=100:00:00

# PAT-217: open-ended (free-generation) HotpotQA eval.
# Renders prompt-only input, runs model.generate(), computes F1/EM vs gold answer.
# Compare results against teacher-forced numbers (baseline ~0.068, motif-infer ~0.657).
#
# Modes:
#   MOTIF_MODE=off     → vanilla checkpoint (no motif), baseline
#   MOTIF_MODE=real    → low-freq residual motif (inference-only, same as teacher-forced real)
#   MOTIF_MODE=nope    → NoPE-only (no residual motif)

CKPT="${CKPT:-/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/rotary_mix_finetune/s42/checkpoint-15900}"
MODE="${MODE:-real}"
PCUT="${PCUT:-512}"
LAYERS="${LAYERS:-0-1-2-3-4-5-6-7-8-9-10-11}"
LENGTHS="${LENGTHS:-2048 4096}"
NCASES="${NCASES:-500}"
MAX_NEW="${MAX_NEW:-50}"
NPROC="${NPROC:-4}"
MODELNAME="${MODELNAME:-free_gen_${MODE}_P${PCUT}}"

_slack() {
    python3 /project/nlp-work5/hongyu-s/gate1/scripts/notify_slack.py \
        --exit-code "$1" --job-id "${SLURM_JOB_ID}" --node "${SLURMD_NODENAME}" \
        --issue "PAT-217" --gpu "4xa6000" \
        --summary "PAT-217 free-gen eval ${MODELNAME} L=${LENGTHS}" 2>/dev/null || true
}
trap '_slack $?' EXIT
set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
    set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi
export PYTHONPATH=/cl/work5/hongyu-s/flash-linear-attention:/cl/work5/hongyu-s/transformers/src${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1

BASE=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
OUT_D=${BASE}/hotpot_long/analysis_outputs/distill_bias
JSONL=${BASE}/hotpot_long/data/hotpot_long_dev_uniform.jsonl

export MOTIF_MODE=${MODE}
export MOTIF_PERIOD_CUTOFF=${PCUT}
export MOTIF_LAYERS=${LAYERS}
export MOTIF_LAM=1.0
export MOTIF_LAM_MODE=const
export MOTIF_DIM_SELECTIVE=1
export MOTIF_NPZ=${OUT_D}/distilled_bias_lowfreq_L512.npz

cd "${BASE}"
mkdir -p hotpot_long/logs

for L in ${LENGTHS}; do
    OUT=${BASE}/hotpot_long/analysis_outputs/pat217_motif/${MODELNAME}/free_gen_L${L}
    mkdir -p "${OUT}"
    echo "=== free-gen ${MODELNAME} L=${L} ==="
    MP=$(( 20000 + SLURM_JOB_ID % 5000 + L % 100 ))
    python -m torch.distributed.run --nproc_per_node=${NPROC} --master_port=${MP} \
        hotpot_long/hotpot_free_gen_eval.py \
        --checkpoint "${CKPT}" \
        --jsonl "${JSONL}" \
        --L "${L}" \
        --n_cases "${NCASES}" \
        --max_new_tokens "${MAX_NEW}" \
        --out "${OUT}"
    echo "--- done L=${L} ---"
done
echo "=== Done: PAT-217 free-gen eval ${MODELNAME} ==="

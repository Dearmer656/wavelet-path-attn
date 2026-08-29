#!/bin/bash
#SBATCH --job-name=pat254_preflight
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_pat254_preflight.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:p6000:1
#SBATCH --nodelist=elm81
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# PAT-254 Preflight sweep: K1 rho128/256, small model, seed 42, L in {512,4096}.

_slack() {
    python3 /project/nlp-work5/hongyu-s/gate1/scripts/notify_slack.py \
        --exit-code "$1" \
        --job-id "${SLURM_JOB_ID}" \
        --node "${SLURMD_NODENAME}" \
        --issue "PAT-254" \
        --gpu "6000x1" \
        --summary "PAT-254 Preflight: real wavelet field sweep, K1 rho128/256, L512/4096"
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

RHO128=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat244_dual_temp/K1_L512_me14_rho128_ricker_s42/checkpoint-15000
RHO256=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat244_dual_temp/K1_L512_me16_rho256_ricker_s42/checkpoint-15000
SCRIPT=analysis/qwab_mechanism_top10/task_00_preflight/preflight_wavelet_field.py

for CKPT_TAG in "${RHO128} small_K1rho128_s42" "${RHO256} small_K1rho256_s42"; do
  CKPT=$(echo "${CKPT_TAG}" | cut -d' ' -f1)
  TAG=$(echo "${CKPT_TAG}" | cut -d' ' -f2)
  for L in 512 4096; do
    python "${SCRIPT}" --checkpoint "${CKPT}" --model_tag "${TAG}" --seq_len "${L}" --n_case 10 --n_qgrid 8
  done
done

echo "=== DONE: PAT-254 Preflight sweep ==="

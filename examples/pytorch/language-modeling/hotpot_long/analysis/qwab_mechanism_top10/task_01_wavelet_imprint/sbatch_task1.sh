#!/bin/bash
#SBATCH --job-name=pat254_task1
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_pat254_task1.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:p6000:1
#SBATCH --nodelist=elm81
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# PAT-254 Task 1: wavelet-subspace imprint, K1 rho128, small model, seed 42,
# L in {512,4096}.

_slack() {
    python3 /project/nlp-work5/hongyu-s/gate1/scripts/notify_slack.py \
        --exit-code "$1" \
        --job-id "${SLURM_JOB_ID}" \
        --node "${SLURMD_NODENAME}" \
        --issue "PAT-254" \
        --gpu "6000x1" \
        --summary "PAT-254 Task 1: wavelet imprint R_wav, K1 rho128 s42, L512/4096"
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

PA=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/PA_baseline_multi_seeds/token_even_mix_PA_s42/checkpoint-15000
QWAB=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat244_dual_temp/K1_L512_me14_rho128_ricker_s42/checkpoint-15000
SCRIPT=analysis/qwab_mechanism_top10/task_01_wavelet_imprint/task1_wavelet_imprint.py

for L in 512 4096; do
  python "${SCRIPT}" --pa_checkpoint "${PA}" --qwab_checkpoint "${QWAB}" \
    --model_tag small_K1rho128_s42 --seq_len "${L}" --n_case 10 --n_random_controls 8
done

echo "=== DONE: PAT-254 Task 1 sweep ==="

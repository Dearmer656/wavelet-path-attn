#!/bin/bash
#SBATCH --job-name=pat254_paqoffqon3
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_pat254_paqoffqon3.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:p6000:1
#SBATCH --nodelist=elm81
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# PAT-254: properly-powered re-check of PA/Q-off/Q-on decomposition -- the
# n=30 s42-only read (Delta_train=-0.117) failed a 2-seed replication check
# (s43 gave +0.032). Re-running at n=100, all 3 seeds, L512, to get a stable
# estimate before trusting any downstream parameter-localization story.

_slack() {
    python3 /project/nlp-work5/hongyu-s/gate1/scripts/notify_slack.py \
        --exit-code "$1" --job-id "${SLURM_JOB_ID}" --node "${SLURMD_NODENAME}" \
        --issue "PAT-254" --gpu "p6000x1" \
        --summary "PAT-254: PA/Qoff/Qon decomposition, n=100 x 3 seeds, L512"
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

SCRIPT=analysis/qwab_mechanism_top10/task_00_preflight/pa_qoff_qon_f1_decomposition.py
PA_BASE=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/PA_baseline_multi_seeds
QWAB_BASE=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat244_dual_temp

declare -A PA_DIR=( [42]="token_even_mix_PA_s42" [43]="s43" [44]="s44" )
for SEED in 42 43 44; do
  PA="${PA_BASE}/${PA_DIR[$SEED]}/checkpoint-15000"
  QWAB="${QWAB_BASE}/K1_L512_me14_rho128_ricker_s${SEED}/checkpoint-15000"
  python "${SCRIPT}" --pa_checkpoint "${PA}" --qwab_checkpoint "${QWAB}" \
    --model_tag "small_K1rho128_s${SEED}_n100" --seq_len 512 --n_case 100
done

echo "=== DONE: PAT-254 3-seed PA/Qoff/Qon n=100 L512 ==="

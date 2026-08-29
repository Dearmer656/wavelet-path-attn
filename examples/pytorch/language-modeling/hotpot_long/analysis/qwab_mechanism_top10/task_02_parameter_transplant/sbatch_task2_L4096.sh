#!/bin/bash
#SBATCH --job-name=pat254_task2_L4096
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_pat254_task2_L4096.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:p6000:1
#SBATCH --nodelist=elm81
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# PAT-254 Task 2 at extrapolated length: all 5 groups, L4096, K1 rho128 s42 --
# does the same parameter-group story hold at the length where Delta_train
# shrinks (-0.039) and the online branch partially recovers it?

_slack() {
    python3 /project/nlp-work5/hongyu-s/gate1/scripts/notify_slack.py \
        --exit-code "$1" --job-id "${SLURM_JOB_ID}" --node "${SLURMD_NODENAME}" \
        --issue "PAT-254" --gpu "p6000x1" \
        --summary "PAT-254 Task 2 @ L4096: all groups, K1 rho128 s42"
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

python analysis/qwab_mechanism_top10/task_02_parameter_transplant/task2_param_transplant.py \
  --pa_checkpoint "${PA}" --qwab_checkpoint "${QWAB}" \
  --model_tag small_K1rho128_s42 --seq_len 4096 --n_case 30 --groups H,QKVO,MLP,LN,embed

echo "=== DONE: PAT-254 Task 2 L4096 ==="

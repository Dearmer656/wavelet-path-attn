#!/bin/bash
#SBATCH --job-name=motif_lf_eval
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_motif_lf_eval.txt
#SBATCH --partition=gpu_long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=100:00:00

# PAT-217 x PAT-208: teacher-forced HotpotQA F1 eval of a low-freq motif finetune.
# CKPT must point at the finetuned run (checkpoint-3000). MODE/PCUT/LAYERS must match the FT.
MODE="${MODE:-real}"
LAM="${LAM:-1.0}"
PCUT="${PCUT:-512}"
LAYERS="${LAYERS:-0-1-2-3-4-5-6-7-8-9-10-11}"
LENGTHS="${LENGTHS:-2048 4096}"
CKPT="${CKPT:?set CKPT to the finetuned run dir, e.g. runs/pat217_motif_ft/motif_lf_real_P512/checkpoint-3000}"
MODELNAME="${MODELNAME:-motif_lf_${MODE}_P${PCUT}}"

_slack() {
    python3 /project/nlp-work5/hongyu-s/gate1/scripts/notify_slack.py \
        --exit-code "$1" --job-id "${SLURM_JOB_ID}" --node "${SLURMD_NODENAME}" \
        --issue "PAT-217" --gpu "1xa6000" \
        --summary "PAT-217 low-freq motif eval ${MODELNAME} L=${LENGTHS}" 2>/dev/null || true
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
export MOTIF_NPZ=${OUT_D}/distilled_bias_lowfreq_L512.npz
export MOTIF_MODE=${MODE}
export MOTIF_LAM=${LAM}
export MOTIF_LAYERS=${LAYERS}
export MOTIF_LAM_MODE=const
export MOTIF_DIM_SELECTIVE=1
export MOTIF_PERIOD_CUTOFF=${PCUT}

cd "${BASE}"
JSONL=${BASE}/hotpot_long/data/hotpot_long_dev_uniform.jsonl
for L in ${LENGTHS}; do
  OUT=${BASE}/hotpot_long/analysis_outputs/pat217_motif/${MODELNAME}/L${L}
  mkdir -p "${OUT}"
  echo "=== ${MODELNAME} L${L} ==="
  MP=$(( 19000 + SLURM_JOB_ID % 4000 + L % 100 ))
  python -m torch.distributed.run --nproc_per_node=1 --master_port=${MP} hotpot_long/motif_launcher.py \
    --model_type gpt2 --tokenizer_name gpt2 --model_name_or_path "${CKPT}" \
    --attn_implementation eager --pe_method rotary \
    --dataset_name hotpot_qa --dataset_config_name distractor \
    --hotpot_long_jsonl "${JSONL}" --hotpot_long_lengths "${L}" \
    --do_eval --block_size "${L}" --per_device_eval_batch_size 4 \
    --eval_generate_no_cache True \
    --output_dir "${OUT}" --overwrite_output_dir --logging_dir "${OUT}/log" \
    --seed 42 --load_best_model_at_end False
  python3 -c "import json;d=json.load(open('${OUT}/eval_results.json'));print('[RESULT] ${MODELNAME} L${L}: F1=%.4f EM=%.4f loss=%.4f'%(d.get('eval_f1',-1),d.get('eval_em',-1),d.get('eval_loss',-1)))" 2>/dev/null || true
done
echo "=== Done: PAT-217 low-freq motif eval ${MODELNAME} ==="

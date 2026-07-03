#!/bin/bash
#SBATCH --job-name=ntk_15900
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_ntk_hotpot_s42_ckpt15900.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:1
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# NTK-aware rotary baseline re-run on checkpoint-15900 (to align with the NoPE/low-freq
# evals, which all use 15900). Same teacher-forced run_clm eval as before; only CKPT changed.
# NTK: theta_new = 10000 * (L/512)^(64/62)  -> L512:10000  L2048:41824  L4096:85498
_slack() {
    python3 /project/nlp-work5/hongyu-s/gate1/scripts/notify_slack.py \
        --exit-code "$1" --job-id "${SLURM_JOB_ID}" --node "${SLURMD_NODENAME}" \
        --issue "PAT-208" --gpu "1x6000" \
        --summary "NTK baseline re-run on ckpt15900 (align with NoPE evals)" 2>/dev/null || true
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
CKPT="${BASE}/runs/rotary_mix_finetune/s42/checkpoint-15900"
JSONL="${BASE}/hotpot_long/data/hotpot_long_dev_uniform.jsonl"
cd "${BASE}"

declare -a LENGTHS=(512   2048   4096)
declare -a THETAS=(10000  41824  85498)

for i in "${!LENGTHS[@]}"; do
  L="${LENGTHS[$i]}"
  THETA="${THETAS[$i]}"
  OUTPUT="${BASE}/hotpot_long/results_uniform/rotary_ntk_s42_ckpt15900_fixed/L${L}"
  mkdir -p "${OUTPUT}"
  echo "=== Rotary NTK (ckpt15900) s42 L${L} theta=${THETA} ==="
  MASTER_PORT=$(( 18000 + SLURM_JOB_ID % 1000 + L % 100 ))
  python -m torch.distributed.run --nproc_per_node=1 --master_port=${MASTER_PORT} ./run_clm.py \
    --model_type gpt2 --tokenizer_name gpt2 \
    --model_name_or_path "${CKPT}" \
    --attn_implementation eager \
    --pe_method rotary \
    --rope_theta "${THETA}" \
    --dataset_name hotpot_qa --dataset_config_name distractor \
    --hotpot_long_jsonl "${JSONL}" \
    --hotpot_long_lengths "${L}" \
    --do_eval \
    --block_size "${L}" \
    --per_device_eval_batch_size 4 \
    --output_dir "${OUTPUT}" --overwrite_output_dir \
    --logging_dir "${OUTPUT}/log" \
    --seed 42 --load_best_model_at_end False
  python3 -c "import json; d=json.load(open('${OUTPUT}/eval_results.json')); print(f'[RESULT] NTK ckpt15900 L${L}: F1={d[\"eval_f1\"]:.4f} EM={d[\"eval_em\"]:.4f} loss={d[\"eval_loss\"]:.4f}')"
done
echo "=== Done: NTK (ckpt15900) s42 HotpotQA ==="

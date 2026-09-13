#!/bin/bash
#SBATCH --job-name=fox_med_forgetgate_perhead
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/log_file/train/%j_fox_medium_forgetgate_perhead.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=25:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# Per-HEAD (not just per-layer-mean) forget-gate retention breakdown, same checkpoint/probe
# as analyze_fox_medium_forget_gate.sh (job 587627). The original dump collapsed the [B,T,H]
# retention tensor's mean over ALL dims including heads, which hides any small subset of
# near-1.0-retention "persistent" heads that could carry HotpotQA-Long's long-range signal
# even while the population mean looks like a ~1-6 token half-life. Rerunning with
# run_clm.py's extended dump_forget_gate_json (now also reports per-layer per-head means
# and the half-life implied by each layer's single highest-retention head).

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
cd "${WORKDIR}"

CKPT="${WORKDIR}/runs/mix_medium_owt_fox_10ep_s42_a6000x4/checkpoint-15000"
[ -d "${CKPT}" ] || { echo "Missing ${CKPT}" >&2; exit 1; }

DATA_DIR="${WORKDIR}/hotpot_long/data"
RESULT_BASE="${WORKDIR}/hotpot_long/results/fox_medium_s42_finetuned_ckpt15000_forgetgate"
mkdir -p "${RESULT_BASE}"
DUMP_JSON="${RESULT_BASE}/forget_gate_per_layer_perhead_L4096.json"

CFG_PATH="${RESULT_BASE}/supply_model_perhead.cfg"
cat > "${CFG_PATH}" <<CFG
dump_forget_gate_json="${DUMP_JSON}"
CFG

OUT_DIR="${RESULT_BASE}/run_perhead"
mkdir -p "${OUT_DIR}"

echo "=== FoX medium (finetuned) ckpt-15000: PER-HEAD forget-gate retention on HotpotQA-Long L4096 (1x a6000) ==="

python ./run_clm.py \
  --model_type forgetting_transformer --tokenizer_name gpt2 \
  --model_name_or_path "${CKPT}" \
  --dataset_name hotpot_qa --dataset_config_name distractor \
  --hotpot_long_jsonl "${DATA_DIR}/hotpot_long_dev.jsonl" \
  --hotpot_long_lengths 4096 \
  --do_eval \
  --max_eval_samples 100 \
  --block_size 4096 \
  --per_device_eval_batch_size 1 \
  --output_dir "${OUT_DIR}" --overwrite_output_dir \
  --logging_dir "${OUT_DIR}/log" \
  --seed 42 --load_best_model_at_end False \
  --cfg_path "${CFG_PATH}"

echo "=== Done. Per-head forget-gate table: ${DUMP_JSON} ==="
cat "${DUMP_JSON}" 2>/dev/null || echo "[WARN] ${DUMP_JSON} not found -- check log for [ForgetGateDump] lines / errors."

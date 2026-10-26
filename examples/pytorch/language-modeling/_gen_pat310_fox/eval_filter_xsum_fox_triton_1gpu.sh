#!/bin/bash
#SBATCH --job-name=xsum_flt_eval
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/head_wise_scale_selection_vs_layer_wise/layer_wise/sigmoid_exp/eval_log/xsum_filter_%j.txt
#SBATCH --partition=gpu_long
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a6000:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# Usage: sbatch eval_filter_xsum_ckpt15000.sh <RUN_DIR> <WAVELET_MODE> <SEED> <MODEL_LABEL>
# WAVELET_MODE: db1 for PA, logit_bias_ctxscale_shift_v0 for WR
# Example (PA s42):
#   sbatch eval_filter_xsum_ckpt15000.sh \
#     .../runs/PA_baseline_multi_seeds/token_even_mix_PA_s42 db1 42 PA_s42

set -euxo pipefail

RUN_DIR="${1:?RUN_DIR required}"
WAVELET_MODE="${2:?WAVELET_MODE required}"
SEED="${3:?SEED required}"
MODEL_LABEL="${4:?MODEL_LABEL required}"

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u
  source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh
  conda activate latest_transformers
  set -u
fi

WORKDIR=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
cd "${WORKDIR}"
export PYTHONPATH=/cl/work5/hongyu-s/transformers/src:${PYTHONPATH:-}
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/.cache/huggingface/datasets
export HF_HOME=/cl/work5/hongyu-s/.cache/huggingface
export XSUM_ROUGE_ONLY=1

if [ -n "${SLURM_NODELIST:-}" ]; then
  export MASTER_ADDR="$(scontrol show hostname "${SLURM_NODELIST}" | head -n 1)"
  export WORLD_SIZE="${SLURM_NTASKS:-1}"
  export RANK="${SLURM_PROCID:-0}"
fi

CKPT_DIR="${RUN_DIR}/checkpoint-15000"
CFG_PATH="${RUN_DIR}/supply_model.cfg"
BIAS_TYPE="$(grep -oP '(?<=bias_type=)"?[a-zA-Z_]+' "${CFG_PATH}" | tr -d '"' | head -1 || true)"
BIAS_TYPE="${BIAS_TYPE:-wavelet}"
# PA-only checkpoints have no wavelet_ctxscale_k in their cfg, so modeling_gpt2.py
# falls back to --router_band_num (8, passed below for an unrelated router feature)
# as K, which then fails validation against the scalar wavelet_ctxscale_scale_max_exp
# default. Pin K=1 via a per-run cfg copy (not the shared original) when running PA-only.
if [ "${WAVELET_MODE}" = "db1" ] && ! grep -q "wavelet_ctxscale_k" "${CFG_PATH}"; then
  PATCHED_CFG="/tmp/supply_model_${MODEL_LABEL}_${SEED}_k1.cfg"
  cp "${CFG_PATH}" "${PATCHED_CFG}"
  printf '\nwavelet_ctxscale_k=1\n' >> "${PATCHED_CFG}"
  CFG_PATH="${PATCHED_CFG}"
fi
XSUM_VALIDATION_FILE="/cl/work5/hongyu-s/fact-check-summarization/xsum_test_filter_level2_official_style.jsonl"
RUN_TAG="${SLURM_JOB_ID:-manual}_$(date -u +%Y%m%dT%H%M%SZ)"

RESULT_DIR="${RUN_DIR}/ckpt_eval_hm"
OUT_ROOT_DIR="${RESULT_DIR}/latest_eval_out_${RUN_TAG}"
XSUM_CSV="${RESULT_DIR}/xsum_filter_metrics_by_len_${MODEL_LABEL}_${RUN_TAG}.csv"
XSUM_PLOT="${RESULT_DIR}/xsum_filter_bertscore_${MODEL_LABEL}_${RUN_TAG}.png"

mkdir -p "${RESULT_DIR}" "${OUT_ROOT_DIR}"

if [ ! -f "${XSUM_CSV}" ]; then
  echo "step,checkpoint,block_size,batch_size,bucket_size,bertscore,bleu,sacrebleu,meteor,chrf,rouge1,rouge2,rougeL,gen_len,ref_len,compression_ratio,empty_pred_ratio,count,perplexity,eval_loss,timestamp_utc" > "${XSUM_CSV}"
fi

declare -a XSUM_BLOCK_SIZES=(512 1024 1536)
declare -a XSUM_BATCH_SIZES=(16   8    4)
declare -a XSUM_BUCKET_SIZES=(512 512  512)

extract_metric() {
  local key="$1"
  local file="$2"
  awk -F: -v target="\"${key}\"" '$1 ~ target {gsub(/[ ,]/, "", $2); gsub(/"/, "", $2); print $2; exit}' "${file}"
}

extract_first_metric() {
  local file="$1"
  shift
  local key="" value=""
  for key in "$@"; do
    value="$(extract_metric "${key}" "${file}")"
    if [ -n "${value}" ]; then echo "${value}"; return 0; fi
  done
  echo ""
}

render_plot() {
  local csv_path="$1" plot_path="$2" y_key="$3" y_label="$4" title="$5"
  python - "${csv_path}" "${plot_path}" "${y_key}" "${y_label}" "${title}" <<'PY'
import csv, sys
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
csv_path, plot_path, y_key, y_label, title = sys.argv[1:]
series = defaultdict(list)
with open(csv_path, "r", encoding="utf-8", newline="") as fin:
    reader = csv.DictReader(fin)
    for row in reader:
        try:
            step = int(row["step"]); y = float(row[y_key])
        except (ValueError, KeyError):
            continue
        series[f"L{row.get('block_size','?')}"].append((step, y))
if not series: sys.exit(0)
plt.figure(figsize=(8.5, 5.2))
for label, points in sorted(series.items()):
    points = sorted(points, key=lambda x: x[0])
    plt.plot([p[0] for p in points], [p[1] for p in points], marker="o", linewidth=1.5, label=label)
plt.xlabel("Steps"); plt.ylabel(y_label); plt.title(title)
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(plot_path, dpi=180)
PY
}

JOB_PORT_OFFSET=$(( ${SLURM_JOB_ID:-0} % 1000 ))
XSUM_PORT=$((19021 + JOB_PORT_OFFSET))

echo "=== Model: ${MODEL_LABEL} | ckpt: ${CKPT_DIR} | wavelet_mode: ${WAVELET_MODE} | seed: ${SEED} ==="

for i in "${!XSUM_BLOCK_SIZES[@]}"; do
  block_size="${XSUM_BLOCK_SIZES[$i]}"
  batch_size="${XSUM_BATCH_SIZES[$i]}"
  bucket_size="${XSUM_BUCKET_SIZES[$i]}"
  OUT_DIR="${OUT_ROOT_DIR}/bs_${block_size}"
  mkdir -p "${OUT_DIR}"
  echo "  -> block_size=${block_size} batch=${batch_size} bucket=${bucket_size}"

  python -m torch.distributed.run --nproc_per_node=1 --master_port="${XSUM_PORT}" ./run_clm.py \
    --model_type gpt2 \
    --tokenizer_name gpt2 \
    --model_name_or_path "${CKPT_DIR}" \
    --bias_type "${BIAS_TYPE}" \
    --share_freq_across_heads True \
    --learning_rate 1e-4 \
    --per_device_train_batch_size "${batch_size}" \
    --per_device_eval_batch_size "${batch_size}" \
    --block_size "${block_size}" \
    --dataset_name xsum \
    --dataset_config_name default \
    --validation_file "${XSUM_VALIDATION_FILE}" \
    --do_eval \
    --num_train_epochs 30 \
    --output_dir "${OUT_DIR}" \
    --eval_strategy steps \
    --logging_dir "${RESULT_DIR}" \
    --logging_steps 250 \
    --save_steps 2500 \
    --attn_implementation path_attn \
    --path_use_qk_norm false \
    --path_use_low_rank_w true \
    --path_use_w_shortconv false \
    --path_conv_size 3 \
    --warmup_steps 5000 \
    --path_conv_bias false \
    --gradient_accumulation_steps 1 \
    --num_harmonics 1 \
    --single_A_B True \
    --use_beta_modulation False \
    --use_soft_wavelet_fox False \
    --wavelet_mode "${WAVELET_MODE}" \
    --wavelet_baseline_use False \
    --init_theta 0.847 \
    --use_forget_gate True --path_attn_impl triton \
    --xsum_bucket_size "${bucket_size}" \
    --xsum_bucket_apply_to eval_test \
    --spectral_loss_coe 0.1 \
    --temp_loss_coe 0.0 \
    --scale_range 0 16 \
    --qk_rotation False \
    --ablate_switch False \
    --wavelet_router False \
    --router_hidden_dim 32 \
    --router_band_num 8 \
    --rel_selection all \
    --cfg_path "${CFG_PATH}" \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --greater_is_better False \
    --seed "${SEED}"

  eval_results_file="${OUT_DIR}/eval_results.json"
  if [ ! -f "${eval_results_file}" ]; then
    echo "[WARN] Missing ${eval_results_file}; skip block=${block_size}"
    continue
  fi

  bertscore="$(extract_first_metric "${eval_results_file}" eval_bertscore bertscore)"
  bleu="$(extract_first_metric "${eval_results_file}" eval_bleu bleu)"
  sacrebleu="$(extract_first_metric "${eval_results_file}" eval_sacrebleu sacrebleu)"
  meteor="$(extract_first_metric "${eval_results_file}" eval_meteor meteor)"
  chrf="$(extract_first_metric "${eval_results_file}" eval_chrf chrf)"
  rouge1="$(extract_first_metric "${eval_results_file}" eval_rouge1 rouge1)"
  rouge2="$(extract_first_metric "${eval_results_file}" eval_rouge2 rouge2)"
  rougeL="$(extract_first_metric "${eval_results_file}" eval_rougeL rougeL)"
  gen_len="$(extract_first_metric "${eval_results_file}" eval_gen_len gen_len)"
  ref_len="$(extract_first_metric "${eval_results_file}" eval_ref_len ref_len)"
  compression_ratio="$(extract_first_metric "${eval_results_file}" eval_compression_ratio compression_ratio)"
  empty_pred_ratio="$(extract_first_metric "${eval_results_file}" eval_empty_pred_ratio empty_pred_ratio)"
  count="$(extract_first_metric "${eval_results_file}" eval_count count)"
  perplexity="$(extract_metric "perplexity" "${eval_results_file}")"
  [ -z "${perplexity}" ] && perplexity="$(extract_metric "eval_perplexity" "${eval_results_file}")"
  eval_loss="$(extract_first_metric "${eval_results_file}" eval_loss loss)"
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  : "${bertscore:=nan}" "${bleu:=nan}" "${sacrebleu:=nan}" "${meteor:=nan}" "${chrf:=nan}"
  : "${rouge1:=nan}" "${rouge2:=nan}" "${rougeL:=nan}" "${gen_len:=nan}" "${ref_len:=nan}"
  : "${compression_ratio:=nan}" "${empty_pred_ratio:=nan}" "${count:=nan}" "${perplexity:=nan}" "${eval_loss:=nan}"

  echo "15000,${CKPT_DIR},${block_size},${batch_size},${bucket_size},${bertscore},${bleu},${sacrebleu},${meteor},${chrf},${rouge1},${rouge2},${rougeL},${gen_len},${ref_len},${compression_ratio},${empty_pred_ratio},${count},${perplexity},${eval_loss},${ts}" >> "${XSUM_CSV}"
done

render_plot "${XSUM_CSV}" "${XSUM_PLOT}" "bertscore" "BERTScore" "${MODEL_LABEL} XSum Filtered BERTScore by Block Size"

echo "Done. CSV: ${XSUM_CSV}"

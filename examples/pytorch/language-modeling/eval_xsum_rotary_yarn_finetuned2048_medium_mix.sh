#!/bin/bash
#SBATCH --job-name=xsum_yarn2048_med
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/mix_medium_owt_rotary_yarn_2048_s42_fp32/ckpt_eval_xsum/%j_xsum_rotary_yarn_finetuned2048_medium_mix.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:4
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Filtered XSum eval for the GENUINELY YaRN-finetuned Rotary GPT-2 medium checkpoint
# (mix_medium_owt_rotary_yarn_2048_s42_fp32/checkpoint-400, tuned at target length 2048
# with yarn_factor=4). Mirrors eval_filter_xsum_ckpt15900_WR.sh's block_size/batch_size
# grid exactly for direct comparability with QWAB: (512,1024,1536) with batch (16,8,4).
# yarn_factor is kept FIXED at 4.0 (matching the value this checkpoint's weights were
# actually adapted to) rather than recomputed per block_size -- all XSum block_sizes here
# (512/1024/1536) are shorter than the 2048 the model was tuned for, so this is an
# interpolation regime under the SAME frequency scaling the model saw during finetuning,
# not a fresh extrapolation config never seen in training.
# pe_method=rotary, attn_implementation=eager (required for the YaRN q/k scaling path).
# Depends on the mix-finetune (train_gpt2_medium_owt_mix_rotary_yarn_2048_s42_fp32.sh,
# job 577962, already completed) -- can be submitted directly, no dependency needed.

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export SKIP_FENICE=1
export SKIP_SUMMAC=1
export WANDB_DISABLED=true
export WANDB_MODE=disabled

WORKDIR=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
cd "${WORKDIR}"

CKPT="${WORKDIR}/runs/mix_medium_owt_rotary_yarn_2048_s42_fp32/checkpoint-400"
XSUM_VALIDATION_FILE="/cl/work5/hongyu-s/fact-check-summarization/xsum_test_filter_level2_official_style.jsonl"
RESULT_DIR="${WORKDIR}/runs/mix_medium_owt_rotary_yarn_2048_s42_fp32/ckpt_eval_xsum"
RUN_TAG="${SLURM_JOB_ID:-manual}_$(date -u +%Y%m%dT%H%M%SZ)"
OUT_ROOT="${RESULT_DIR}/xsum_out_${RUN_TAG}"
XSUM_CSV="${RESULT_DIR}/xsum_filter_metrics_rotary_yarn_finetuned2048_${RUN_TAG}.csv"
mkdir -p "${RESULT_DIR}" "${OUT_ROOT}"

echo "step,checkpoint,block_size,batch_size,rouge1,rouge2,rougeL,bertscore,count,eval_loss,timestamp_utc" > "${XSUM_CSV}"

declare -a BLOCK_SIZES=(512 1024 1536)
declare -a BATCH_SIZES=(16  8    4)

JOB_PORT=$(( 19900 + ${SLURM_JOB_ID:-0} % 1000 ))

extract_metric() { local key="$1" file="$2"; awk -F: -v t="\"${key}\"" '$1~t{gsub(/[ ,]/,"",$2);gsub(/"/,"",$2);print $2;exit}' "${file}"; }
extract_first_metric() { local file="$1"; shift; local k v; for k in "$@"; do v="$(extract_metric "${k}" "${file}")"; [ -n "${v}" ] && echo "${v}" && return; done; echo ""; }

echo "=== Rotary+YaRN (finetuned@2048) medium mix XSum eval | ckpt: ${CKPT} ==="

for i in "${!BLOCK_SIZES[@]}"; do
  bs="${BLOCK_SIZES[$i]}"
  batch="${BATCH_SIZES[$i]}"
  OUT="${OUT_ROOT}/bs_${bs}"
  mkdir -p "${OUT}"
  echo "  -> block_size=${bs} batch=${batch}"

  python -m torch.distributed.run --nproc_per_node=4 --master_port="${JOB_PORT}" ./run_clm.py \
    --model_type gpt2 \
    --tokenizer_name gpt2 \
    --model_name_or_path "${CKPT}" \
    --pe_method rotary \
    --attn_implementation eager \
    --use_yarn True \
    --yarn_factor 4 \
    --yarn_original_max_position_embeddings 512 \
    --wavelet_router False \
    --router_band_num 8 \
    --scale_range 0 16 \
    --wavelet_mode logit_bias_ctxscale_shift_v0 \
    --wavelet_baseline_use False \
    --use_beta_modulation False \
    --use_soft_wavelet_fox False \
    --single_A_B True \
    --num_harmonics 1 \
    --share_freq_across_heads True \
    --block_size "${bs}" \
    --dataset_name xsum \
    --dataset_config_name default \
    --validation_file "${XSUM_VALIDATION_FILE}" \
    --do_eval \
    --per_device_eval_batch_size "${batch}" \
    --output_dir "${OUT}" \
    --overwrite_output_dir \
    --xsum_bucket_size 512 \
    --xsum_bucket_apply_to eval_test \
    --load_best_model_at_end False \
    --seed 42

  F="${OUT}/eval_results.json"
  [ -f "${F}" ] || { echo "[WARN] missing ${F}"; continue; }
  r1="$(extract_first_metric "${F}" eval_rouge1 rouge1)"
  r2="$(extract_first_metric "${F}" eval_rouge2 rouge2)"
  rL="$(extract_first_metric "${F}" eval_rougeL rougeL)"
  bs_="$(extract_first_metric "${F}" eval_bertscore bertscore)"
  cnt="$(extract_first_metric "${F}" eval_count count)"
  loss="$(extract_first_metric "${F}" eval_loss loss)"
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  : "${r1:=nan}" "${r2:=nan}" "${rL:=nan}" "${bs_:=nan}" "${cnt:=nan}" "${loss:=nan}"
  echo "400,${CKPT},${bs},${batch},${r1},${r2},${rL},${bs_},${cnt},${loss},${ts}" >> "${XSUM_CSV}"
  echo "  [DONE] L${bs}: rouge1=${r1} rougeL=${rL}"
done

echo "=== Done. CSV: ${XSUM_CSV} ==="

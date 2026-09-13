#!/bin/bash
#SBATCH --job-name=qwab_small_ppl_fullset
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/log_file/train/%j_qwab_small_ppl_L512_L2048_fullset.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:p6000:2
#SBATCH --time=25:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# Confound check: the existing QWAB small L512/L2048 ppl numbers
# (scale_weight_100cases_run: ppl=17.33, n=100; L2048: ppl=17.17, n=100) used
# --max_eval_samples 100, which under-shoots the true dataset size at these two
# lengths (WikiText-103 validation gives 481 blocks at L=512, 119 at L=2048).
# PA-only's own sweep (eval_ppl_PA_only_small_wt103_sweep.sh) used
# --max_eval_samples 2000, which is NOT binding at these lengths, so PA-only's
# reported numbers (ppl=19.29 @ n=481, ppl=17.72 @ n=119) are true full-dataset
# averages while QWAB's are 100-block subsets -- not apples-to-apples. This
# reruns QWAB at the same --max_eval_samples 2000 (non-binding here too) so both
# sides are the true full-dataset average before trusting the L512 ppl gap
# (currently reported as -1.956 nats, by far the largest gap of any length).
# L4096/L8192 are unaffected (both sides already hit the natural dataset-size
# ceiling identically: n=59/26 on both).

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
cd "${WORKDIR}"

CKPT="${WORKDIR}/runs/gpt2_small_wt103_qwab_pathattn_fromscratch_rho256_headshare_80k/checkpoint-80000"
[ -d "${CKPT}" ] || { echo "Missing ${CKPT}" >&2; exit 1; }

RESULT_BASE="${WORKDIR}/hotpot_long/results_uniform/qwab_small_rho256_headshare_ckpt80000_wt103_ppl"
mkdir -p "${RESULT_BASE}"

for L in 512 2048; do
  OUT_DIR="${RESULT_BASE}/L${L}_fullset"
  mkdir -p "${OUT_DIR}"
  DUMP_JSON="${RESULT_BASE}/scale_weight_per_layer_fullset_L${L}.json"

  CFG_PATH="${RESULT_BASE}/supply_model_L${L}_fullset.cfg"
  cat > "${CFG_PATH}" <<CFG
wavelet_ctxscale_k=1
wavelet_ctxscale_scale_max_exp=16.0
wavelet_mode="logit_bias_ctxscale_shift_v0"
dump_scale_weight_json="${DUMP_JSON}"
path_attn_capture_debug_tensors=false
CFG

  MASTER_PORT=$(( 35500 + SLURM_JOB_ID % 1000 + L % 1000 ))
  echo "--- QWAB SMALL ckpt-80000: L=${L} FULL DATASET (max_eval_samples=2000, 2x p6000) ---"
  /cl/work5/hongyu-s/conda/envs/latest_transformers/bin/torchrun \
    --nproc_per_node=2 \
    --master_port="${MASTER_PORT}" \
    ./run_clm.py \
    --model_type gpt2 \
    --tokenizer_name gpt2 \
    --model_name_or_path "${CKPT}" \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --block_size "${L}" \
    --do_eval \
    --max_eval_samples 2000 \
    --per_device_eval_batch_size 1 \
    --attn_implementation path_attn \
    --path_use_qk_norm false \
    --path_use_low_rank_w true \
    --path_use_w_shortconv false \
    --path_conv_size 3 \
    --path_conv_bias false \
    --single_A_B True \
    --share_freq_across_heads True \
    --pe_method vanilla \
    --num_harmonics 1 \
    --wavelet_baseline_use False \
    --wavelet_router False \
    --use_beta_modulation False \
    --use_soft_wavelet_fox False \
    --use_forget_gate False \
    --init_theta 0.847 \
    --scale_range 0 16 \
    --weight_alpha 0.0 \
    --loss_type cos \
    --qk_rotation False \
    --router_band_num 8 \
    --router_hidden_dim 32 \
    --rel_selection all \
    --preprocessing_num_workers 8 \
    --ddp_timeout 21600 \
    --seed 42 \
    --overwrite_output_dir \
    --output_dir "${OUT_DIR}" \
    --logging_dir "${OUT_DIR}/log" \
    --load_best_model_at_end False \
    --cfg_path "${CFG_PATH}"

  echo "L=${L} done: $(cat ${OUT_DIR}/eval_results.json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'eval_loss={d[\"eval_loss\"]:.4f} perplexity={d[\"perplexity\"]:.4f} n={d[\"eval_samples\"]}')" 2>/dev/null || echo 'no results')"
done

echo "=== QWAB small L512/L2048 FULL DATASET rerun done ==="

#!/bin/bash
#SBATCH --job-name=pa_only_ppl_sweep_triton
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/log_file/train/%j_pa_only_ppl_sweep_triton.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:4
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# Continuation of eval_ppl_PA_only_medium_owt_sweep.sh (job 584483): that run
# completed L=512/2048/4096 fine on pytorch, then OOM'd at L=8192 (46.70GB
# already in use, needed 4GB more) -- matches the project's established
# "medium L>=8192 pytorch OOMs, use triton" convention (same boundary QWAB's
# separate OOM at L=4096 does NOT match, since QWAB's wavelet_QH kernel is
# more expensive). Reruns L=8192/12288/16384 with --path_attn_impl triton.

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

CKPT="${WORKDIR}/runs/gpt2_medium_owt_pytorch_level_path_attn/checkpoint-80000"
[ -d "${CKPT}" ] || { echo "Missing ${CKPT}" >&2; exit 1; }

RESULT_BASE="${WORKDIR}/hotpot_long/results_uniform/pa_only_medium_ckpt80000_owt_ppl_triton"
mkdir -p "${RESULT_BASE}"

CFG_PATH="${RESULT_BASE}/supply_model.cfg"
cat > "${CFG_PATH}" <<'CFG'
wavelet_ctxscale_k=1
wavelet_ctxscale_scale_max_exp=16.0
CFG

echo "=== PA-only medium ckpt-80000: OpenWebText perplexity sweep [TRITON, L>=8192] ==="

for L in 8192 12288 16384; do
  OUT_DIR="${RESULT_BASE}/L${L}"
  mkdir -p "${OUT_DIR}"
  MASTER_PORT=$(( 26000 + SLURM_JOB_ID % 1000 + L % 1000 ))
  echo "--- L=${L} ---"
  /cl/work5/hongyu-s/conda/envs/latest_transformers/bin/torchrun \
    --nproc_per_node=4 \
    --master_port="${MASTER_PORT}" \
    ./run_clm.py \
    --model_type gpt2 \
    --tokenizer_name gpt2 \
    --model_name_or_path "${CKPT}" \
    --dataset_name openwebtext \
    --validation_split_percentage 1 \
    --block_size "${L}" \
    --do_eval \
    --max_eval_samples 2000 \
    --per_device_eval_batch_size 1 \
    --attn_implementation path_attn \
    --path_attn_impl triton \
    --path_use_qk_norm false \
    --path_use_low_rank_w true \
    --path_use_w_shortconv false \
    --path_conv_size 3 \
    --path_conv_bias false \
    --single_A_B True \
    --share_freq_across_heads True \
    --pe_method vanilla \
    --num_harmonics 1 \
    --wavelet_mode db1 \
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
  echo "L=${L} done: $(cat ${OUT_DIR}/eval_results.json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'eval_loss={d[\"eval_loss\"]:.4f} perplexity={d[\"perplexity\"]:.4f}')" 2>/dev/null || echo 'no results')"
done

echo "=== ALL TRITON LENGTHS DONE ==="
for L in 8192 12288 16384; do
  OUT_DIR="${RESULT_BASE}/L${L}"
  echo "L=${L}: $(cat ${OUT_DIR}/eval_results.json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'eval_loss={d[\"eval_loss\"]:.4f} perplexity={d[\"perplexity\"]:.4f}')" 2>/dev/null || echo 'no results')"
done

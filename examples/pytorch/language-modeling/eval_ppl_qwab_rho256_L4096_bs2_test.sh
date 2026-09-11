#!/bin/bash
#SBATCH --job-name=qwab_L4096_bs2_test
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/log_file/train/%j_qwab_L4096_bs2_test.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:p6000:1
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# Batch-size headroom probe: same config as eval_ppl_qwab_rho256_L4096_pytorch_debugoff_test.sh
# (job 584902, currently running at bs=1 on the other free p6000 GPU -- left
# untouched), but per_device_eval_batch_size=2 on only 50 samples (just need
# to see whether it OOMs or not, not a full eval).

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

CKPT="${WORKDIR}/runs/gpt2_medium_owt_qwab_pathattn_fromscratch_rho256_80k_elm73resume10000/checkpoint-80000"
OUT_DIR="${WORKDIR}/hotpot_long/results_uniform/qwab_medium_rho256_scratch_ckpt80000_owt_ppl_debugoff_test/L4096_bs2_probe"
mkdir -p "${OUT_DIR}"

CFG_PATH="${OUT_DIR}/supply_model.cfg"
cat > "${CFG_PATH}" <<'CFG'
wavelet_ctxscale_k=1
wavelet_ctxscale_scale_max_exp=16.0
path_attn_capture_debug_tensors=false
CFG

MASTER_PORT=$(( 28000 + SLURM_JOB_ID % 1000 ))

echo "=== QWAB rho256 L=4096 pytorch [bs=2 headroom probe] ==="
nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv

/cl/work5/hongyu-s/conda/envs/latest_transformers/bin/torchrun \
  --nproc_per_node=1 \
  --master_port="${MASTER_PORT}" \
  ./run_clm.py \
  --model_type gpt2 \
  --tokenizer_name gpt2 \
  --model_name_or_path "${CKPT}" \
  --dataset_name openwebtext \
  --validation_split_percentage 1 \
  --block_size 4096 \
  --do_eval \
  --max_eval_samples 50 \
  --per_device_eval_batch_size 2 \
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
  --wavelet_pe_softmax_use False \
  --wavelet_mode logit_bias_ctxscale_shift_v0 \
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

echo "=== bs=2 probe done: $(cat ${OUT_DIR}/eval_results.json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'eval_loss={d[\"eval_loss\"]:.4f} perplexity={d[\"perplexity\"]:.4f}')" 2>/dev/null || echo 'no results') ==="

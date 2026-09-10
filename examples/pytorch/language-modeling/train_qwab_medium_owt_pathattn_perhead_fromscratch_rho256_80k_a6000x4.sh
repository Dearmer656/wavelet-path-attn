#!/bin/bash
#SBATCH --job-name=qwab_med_perhead_rho256
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/log_file/train/%j_qwab_medium_owt_pathattn_perhead_fromscratch_rho256_80k_a6000x4.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:4
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# PAT-226: medium-scale per-head QWAB scale-weight router, FROM-SCRATCH pretrain, rho=256.
# Direct per-head counterpart to the currently-running head-shared medium pretrain
# (job 581523, runs/gpt2_medium_owt_qwab_pathattn_fromscratch_rho256_80k_elm73resume10000)
# -- every other setting kept identical to that run's own cfg/CLI:
#   - wavelet_ctxscale_k=1, wavelet_ctxscale_scale_max_exp=16.0 (rho=256), same as
#     581523's own supply_model.cfg (verified by reading that file directly, not assumed)
#   - eval disabled (--eval_strategy no, no --do_eval), matching 581523's no-eval fix
#     (the eval-every-5000-steps overhead finding from this session)
#   - lr=1e-4, weight_decay=0.01, warmup_ratio=0.05, bf16, block_size=512, max_steps=80000,
#     save_steps=10000 -- all copied verbatim from 581523's own CLI
#   - bs=8/accum=2 (global_bs=64) -- same budget as 581523 (48GB-class card, matches
#     the earlier finding that QWAB's wavelet branch is NOT O(T) memory, bs=16 OOM'd on
#     6000/a6000 48GB at this block_size)
# ONLY variable changed: wavelet_ctxscale_router_per_head=true. detach_delta is left
# unset (defaults to False = joint/nodetach) -- this ALREADY matches 581523's own cfg
# (which also never sets this flag), so "nodetach" here is not a new setting, it's the
# same default both runs share; confirmed by reading 581523's actual on-disk cfg file
# before writing this one, not assumed from memory.
# GPU: a6000x4 per explicit request (distinct from 581523's elm73 6000x4, no node overlap).
#
# 2026-09-10: bs=8/accum=2 (581523's own budget) OOM'd immediately on all 4 GPUs at step 1
# ("Tried to allocate 786.00 MiB" with 46.64GB already in use). Root cause: the per-head
# router keeps the full head axis through path_attn.py's K-loop (bias_chunk/eff_chunk
# become [B,H,q_len,T] instead of head-shared's [B,q_len,T]) -- for medium's 16 attention
# heads this multiplies the relevant intermediate tensors by roughly that factor, on top of
# the wavelet branch's already-non-O(T) memory profile (see the rho=128/256 from-scratch
# scripts' own bs=16->bs=8 OOM history). Dropped to bs=2/accum=8 (global_bs=64 unchanged).

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
OUT="${WORKDIR}/runs/gpt2_medium_owt_qwab_pathattn_perhead_fromscratch_rho256_80k"
mkdir -p "${OUT}/train"

CFG_PATH="${OUT}/supply_model.cfg"
cat > "${CFG_PATH}" <<'CFG'
wavelet_ctxscale_k=1
wavelet_ctxscale_scale_max_exp=16.0
wavelet_ctxscale_router_per_head=true
CFG

echo "=== supply_model.cfg written to ${CFG_PATH}: ==="
cat "${CFG_PATH}"
echo "=== end cfg ==="

MASTER_PORT=$(( 24970 + SLURM_JOB_ID % 1000 ))

echo "=== QWAB medium PER-HEAD FROM-SCRATCH OWT pretrain, rho=256, nodetach(default): 4x a6000 ==="

/cl/work5/hongyu-s/conda/envs/latest_transformers/bin/torchrun \
  --nproc_per_node=4 \
  --master_port="${MASTER_PORT}" \
  ./run_clm.py \
  --model_type gpt2 \
  --tokenizer_name gpt2 \
  --config_name openai-community/gpt2-medium \
  --dataset_name openwebtext \
  --validation_split_percentage 1 \
  --block_size 512 \
  --do_train \
  --max_steps 80000 \
  --eval_strategy no \
  --save_steps 10000 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --warmup_ratio 0.05 \
  --bf16 True \
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
  --output_dir "${OUT}" \
  --logging_dir "${OUT}/train/tensorboard" \
  --cfg_path "${CFG_PATH}"

echo "=== QWAB medium per-head from-scratch OWT pretrain (rho=256, a6000x4) done ==="

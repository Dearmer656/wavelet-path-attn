#!/bin/bash
#SBATCH --job-name=qwab_med_scratch
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/log_file/train/%j_qwab_medium_owt_pathattn_fromscratch_80k.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:p6000:4
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# PAT-226: QWAB (wavelet ctxscale bias, "headline" mechanism -- same wavelet_mode as
# mix_medium_owt_dd_10ep) FROM-SCRATCH medium pretrain on OpenWebText.
# Closes the confound flagged in project memory: QWAB (small+medium) has so far only ever
# been a mix-dataset FINETUNE on top of an already-converged plain PaTH-only checkpoint --
# it has never had its own from-scratch pretrain with the wavelet mechanism active, unlike
# Rotary/ALiBi/Mamba-2 which all do. This is that pretrain.
#
# Recipe: kept UNIFORM with every other baseline's pretrain (Rotary/ALiBi/Mamba2's
# cross-baseline convention) rather than QWAB's own finetune-stage recipe, per explicit
# discussion -- lr=1e-4, weight_decay=0.01, warmup_ratio=0.05, bf16, block_size=512,
# max_steps=80000, eval_steps=5000, load_best_model_at_end -- so this varies ONLY
# architecture (path_attn + wavelet ctxscale bias) vs the other pretrains, preserving the
# "identical data" controlled comparison.
#
# Distillation REMOVED (not just disabled): the original dd_10ep finetune recipe uses
# distill_teacher=wavelet (a teacher forward pass distilling into the wavelet branch),
# which only makes sense as a finetuning aid on an already-converged backbone -- at
# from-scratch step 0 the model is randomly initialized, so a "teacher" signal from it is
# meaningless and could destabilize early training. Per explicit request, all
# distillation-related flags (distill_teacher, distill_in_which_layers, distill_freq_scale,
# distilling_coe_warmup_use, spectral_loss_coe, temp_loss_coe, sample_num, smooth_use) are
# simply omitted here (the underlying code stays in run_clm.py, untouched -- 538 other
# scripts still reference these flags; only this script drops them).
#
# wavelet_ctxscale_scale_max_exp supplied explicitly (list of 8) via --cfg_path: the
# codebase's own defaults (wavelet_ctxscale_k=8, scale_max_exp=14.0 scalar) fail the
# codebase's own validation otherwise -- same fix needed for the dd_10ep-based extra-adapt
# finetune script and its ppl quick-check.
#
# 2026-09-07: switched to 4xa100-80 (80GB each) per request, bs 4/accum4 -> 16/accum1
# (global_bs=64 unchanged) -- PaTH attention is O(T) memory (not O(T^2) like eager), and
# 80GB has ~1.7x the headroom of the 6000's 48GB, so this is expected to fit; untested,
# will show as an OOM quickly if not.
# 2026-09-07 (later): a100-80 (elm44) stayed drained with no ETA; switched to 4xp6000
# (24GB each, elm81/82) per request instead, first at bs=2/accum8 (conservative), then
# per follow-up request back to bs=16/accum1 (global_bs=64 unchanged either way) -- this
# pretrain is block_size=512 only (much shorter than the L4096+ lengths that needed bf16
# fallback for PaTH-only's own ppl quick-check on 48GB a6000), so bs=16 has a reasonable
# chance of fitting on p6000's 24GB; untested at this exact combination, will surface as
# an OOM quickly if not.

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
OUT="${WORKDIR}/runs/gpt2_medium_owt_qwab_pathattn_fromscratch_80k"
mkdir -p "${OUT}/train"

# 2026-09-07: rho=128 requested (switched from an earlier rho=256 attempt), K=1 (single
# scale), bias_type=wavelet (Ricker, the default -- multiplier=1.0). rho = 2^(scale_max_exp/2),
# so rho=128 => scale_max_exp=14.0 (verified via modeling_gpt2.py's
# SCALE_MULTIPLIER_DICT/scale formula, not assumed -- this also happens to be the
# codebase's own scale_max_exp default, i.e. rho=128 is literally the framework's default
# single-scale value).
CFG_PATH="${OUT}/supply_model.cfg"
cat > "${CFG_PATH}" <<'CFG'
wavelet_ctxscale_k=1
wavelet_ctxscale_scale_max_exp=14.0
CFG

MASTER_PORT=$(( 24900 + SLURM_JOB_ID % 1000 ))

echo "=== QWAB (wavelet ctxscale, no distillation) medium FROM-SCRATCH OWT pretrain: 4x6000 ==="

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
  --do_train --do_eval \
  --max_steps 80000 \
  --eval_strategy steps --eval_steps 5000 \
  --save_steps 10000 \
  --load_best_model_at_end True --metric_for_best_model eval_loss --greater_is_better False \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 1 \
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

echo "=== QWAB medium from-scratch OWT pretrain done (4x6000) ==="

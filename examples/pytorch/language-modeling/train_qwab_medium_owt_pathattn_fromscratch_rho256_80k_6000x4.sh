#!/bin/bash
#SBATCH --job-name=qwab_med_scratch_rho256
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/log_file/train/%j_qwab_medium_owt_pathattn_fromscratch_rho256_80k.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:4
#SBATCH --nodelist=elm71
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# PAT-226: QWAB (wavelet ctxscale bias) FROM-SCRATCH medium pretrain on OpenWebText, rho=256
# variant -- same as train_qwab_medium_owt_pathattn_fromscratch_80k.sh (rho=128, currently
# queued for p6000, job 579027) except wavelet_ctxscale_scale_max_exp=16.0 (rho=256, also
# the codebase's rho label convention: scale = 2^(max_exp/2), see modeling_gpt2.py
# QWABBias.__init__ comment "rho=256 from the K1 L512 sweep can be reused directly here" --
# max_exp=16.0 is also this class's own getattr default). Not a replacement for 579027;
# both rho variants are wanted (per explicit request: "不cancel现在的任务").
#
# All non-wavelet flags kept byte-for-byte identical to the PaTH-only medium pretrain
# convention (job 416508 / train_gpt2_medium_owt_rotary_80k_6000x4.sh's path_attn analogue):
# lr=1e-4, weight_decay=0.01, warmup_ratio=0.05, bf16, block_size=512, max_steps=80000,
# eval_steps=5000, load_best_model_at_end, per_device_bs=16/accum=1 (global_bs=64) -- so
# this varies ONLY the wavelet scale (rho) vs the already-queued rho=128 run, and ONLY
# architecture (path_attn + wavelet ctxscale bias) vs the other baselines' pretrains.
#
# GPU: 4x6000 on elm71 (idle, 48GB/GPU) instead of p6000 (elm81/82, 24GB/GPU, currently
# fully occupied by another user).
# 2026-09-08: bs=16/accum=1 OOM'd on 6000's 48GB anyway (job 579695, "Tried to allocate
# 128.00 MiB" with 46.63GB already in use, during the very first forward pass). Root
# cause: QWAB's wavelet ctxscale bias mechanism is NOT O(T) memory like plain path_attn --
# fla/layers/path_attn.py's path_ut_base_raw materializes several O(B,H,T,T) intermediate
# tensors per layer (lower_QK, correction, E_base_raw, etc.) that must be retained for
# backward, so it behaves closer to eager's O(T^2) profile once the wavelet branch is
# active. Dropped to bs=8/accum=2 (global_bs=64 unchanged) to fit.
#
# Distillation REMOVED (not just disabled) -- same reasoning as the rho=128 script: a
# from-scratch run has no converged backbone for a distillation teacher signal to be
# meaningful. All distill_* / spectral_loss_coe / temp_loss_coe / sample_num / smooth_use
# flags omitted; run_clm.py itself is untouched.

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
OUT="${WORKDIR}/runs/gpt2_medium_owt_qwab_pathattn_fromscratch_rho256_80k"
mkdir -p "${OUT}/train"

# rho=256 => scale_max_exp=16.0 (scale = 2^(max_exp/2) * SCALE_MULTIPLIER_DICT["wavelet"]=1.0,
# verified directly against modeling_gpt2.py QWABBias.__init__, not assumed). K=1 (single
# scale), bias_type=wavelet (Ricker, the default -- multiplier=1.0).
CFG_PATH="${OUT}/supply_model.cfg"
cat > "${CFG_PATH}" <<'CFG'
wavelet_ctxscale_k=1
wavelet_ctxscale_scale_max_exp=16.0
CFG

MASTER_PORT=$(( 24950 + SLURM_JOB_ID % 1000 ))

echo "=== QWAB (wavelet ctxscale, no distillation) medium FROM-SCRATCH OWT pretrain, rho=256: 4x6000 ==="

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

echo "=== QWAB medium from-scratch OWT pretrain (rho=256) done (4x6000) ==="

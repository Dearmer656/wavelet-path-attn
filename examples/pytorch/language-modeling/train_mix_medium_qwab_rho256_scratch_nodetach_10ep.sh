#! /bin/bash
#SBATCH --job-name=MedMixQWABrho256NoDetach
#SBATCH --output=log_file/train/%j_mix_medium_qwab_rho256_scratch_nodetach_10ep.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:4
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# PAT-258: Stage-2 mix (HotpotQA+XSum) finetune starting from the QWAB
# rho=256 FROM-SCRATCH pretrain checkpoint (job 581523's final
# checkpoint-80000), NOT the PA-only backbone the standard
# train_gpt2_medium_owt_mix_dd_10ep.sh convention uses -- this is the whole
# point: does pretraining WITH QWAB active (vs. adding it only at finetune
# time) carry through to downstream HotpotQA/XSum, on top of what its raw
# OWT perplexity already showed (PAT-258).
# NODETACH variant: wavelet_ctx_feat_detach_delta=false (joint gradient,
# router/shift-proj sees gradient from the full backbone) -- the existing
# "dd" convention's own cfg actually sets detach_delta=true by default
# ("dd" = detach delta), so this is a deliberate deviation, paired with the
# detach sibling run (train_mix_medium_qwab_rho256_scratch_detach_10ep.sh)
# for a clean detach-vs-nodetach comparison on this specific backbone.
# Everything else mirrors train_gpt2_medium_owt_mix_dd_10ep.sh's cfg
# 1:1 except: model_name_or_path, wavelet_ctxscale_k/scale_max_exp (added
# explicitly here, matching the rho=256 pretrain's own saved config, since
# the "dd" script's cfg doesn't set them and this backbone's pretrain did),
# and detach_delta.

set -euxo pipefail
echo 'Workdir: /project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling'
cd /project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling

set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true
export WANDB_MODE=disabled

OUT=runs/mix_medium_owt_qwab_rho256_scratch_nodetach_10ep
mkdir -p "${OUT}"

CFG_PATH="${OUT}/supply_model.cfg"
cat > "${CFG_PATH}" <<'CFG'
router_mode="seperate"
coe_mode="none"
tau=1
scale_type="none"
hotpot_question_position="later"
wavelet_mode='logit_bias_ctxscale_shift_v0'
wavelet_ctxscale_k=1
wavelet_ctxscale_scale_max_exp=16.0
wavelet_logit_bias_a_init=-2
wavelet_ctxscale_tau=1.0
wavelet_ctxscale_router_rms_eps=1e-6
wavelet_ctxscale_chunk_q=128
wavelet_ctxscale_g_max=0.5
wavelet_ctxscale_g_bias_max=4.0
wavelet_ctxscale_lock_window=200
wavelet_ctxscale_lock_grad_eps=1e-6
wavelet_ctxscale_lock_update_eps=1e-6
wavelet_ctxscale_lock_min_frac=0.5
wavelet_ctxscale_lock_clamp_abs=4.0
wavelet_ctxscale_far_only=false
wavelet_ctxscale_far_min_delta=0
wavelet_ctxscale_head_indices='all'
wavelet_logit_bias_eps=1e-6
wavelet_logit_bias_clamp_enable=true
wavelet_logit_bias_clamp_quantile=0.99
wavelet_logit_bias_clamp_min=0.0
wavelet_logit_bias_clamp_scale=1.0
wavelet_logit_bias_log_every=500
wavelet_logit_bias_log_sample_tokens=64
wavelet_logit_bias_log_sample_heads=4
wavelet_logit_bias_debug_assert=false
rel_use_layer_list=all
wavelet_ctxscale_use_head_gate=false
wavelet_ctxscale_scale_dependent_shift=true
wavelet_ctxscale_shift_unit_max=1.0
wavelet_router_chunk_size=1
wavelet_router_chunk_pool="mean"
wavelet_router_chunk_align="left"
wavelet_router_chunk_share=true
wavelet_ctxscale_disable_layer_gate=true

wavelet_router_sigmoid_mode="with_null"
wavelet_ctx_feat_detach_delta=false
wavelet_ctx_feat_mode="q_minus_qcorr_meanh"
CFG

MASTER_PORT=$(( 23400 + SLURM_JOB_ID % 1000 ))

echo '================= BEGIN RUN (QWAB rho256 scratch backbone, NODETACH) ================='

/cl/work5/hongyu-s/conda/envs/latest_transformers/bin/torchrun \
  --nproc_per_node=4 \
  --master_port="${MASTER_PORT}" \
  ./run_clm.py \
  --model_type gpt2 \
  --tokenizer_name gpt2 \
  --config_name openai-community/gpt2-medium \
  --model_name_or_path runs/gpt2_medium_owt_qwab_pathattn_fromscratch_rho256_80k_elm73resume10000/checkpoint-80000 \
  --dataset_name mix \
  --block_size 512 \
  --do_train \
  --num_train_epochs 10 \
  --logging_steps 500 \
  --save_steps 5000 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --weight_decay 0.0 \
  --warmup_ratio 0.05 \
  --attn_implementation path_attn \
  --path_use_qk_norm false \
  --path_use_low_rank_w true \
  --path_use_w_shortconv false \
  --path_conv_size 3 \
  --path_conv_bias false \
  --single_A_B True \
  --share_freq_across_heads True \
  --b_unfreeze_step 5000 \
  --pe_method vanilla \
  --num_harmonics 1 \
  --wavelet_pe_softmax_use False \
  --wavelet_mode logit_bias_ctxscale_shift_v0 \
  --wavelet_baseline_use False \
  --wavelet_router False \
  --use_beta_modulation False \
  --use_soft_wavelet_fox False \
  --use_forget_gate False \
  --full_fine_tune False \
  --init_theta 0.847 \
  --sample_num 16 \
  --spectral_loss_coe 0.1 \
  --temp_loss_coe 0.0 \
  --distill_teacher wavelet \
  --distill_in_which_layers 0 \
  --distill_freq_scale 25 \
  --smooth_use False \
  --distilling_coe_warmup_use False \
  --scale_range 0 16 \
  --weight_alpha 0.0 \
  --loss_type cos \
  --qk_rotation False \
  --router_band_num 8 \
  --router_hidden_dim 32 \
  --rel_selection all \
  --seed 42 \
  --overwrite_output_dir \
  --output_dir "${OUT}" \
  --logging_dir "./${OUT}_log" \
  --cfg_path "${CFG_PATH}"

echo "=== PAT-258 medium mix QWAB rho256-scratch-backbone NODETACH 10ep done ==="

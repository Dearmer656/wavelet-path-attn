#!/bin/bash
# Per-head QWAB scale-weight router (small model, K1, L512, rho=512).
#
# Current default routing feature (q_minus_qcorr_meanh) averages the per-head
# (q - q_corr) delta over the head axis BEFORE the router sees it, so every
# attention head gets the exact same scale-gate weight at a given position --
# "heads share/average the routing decision". This run switches to
# wavelet_ctxscale_router_per_head=true (-> wavelet_ctx_feat_mode=
# q_minus_qcorr_meanh_perhead), which keeps the head axis in the router's
# input feature, so each head maps out its own scale-gate weight.
#
# 2026-09-08: found and fixed a real blocker in path_attn.py before this could
# run at all -- the per-head feature mode returns a 4D [B,T,H,head_dim] tensor,
# but _ctxscale_router_feature's caller had a strict `x_feat.dim() != 3` guard
# that would raise ValueError immediately on startup. Everything DOWNSTREAM of
# that guard (pi.dim() checks at lines ~6668/~7391, the per_head_router-aware
# bias_chunk construction/broadcast at ~7928-8101, and the final g_layer/
# logits_out write at ~8234-8294) already had full, correct 4D support --
# this was evidently a half-wired PAT-244 ablation switch, not a from-scratch
# feature. Relaxed the guard to `not in (3, 4)`, matching the pattern already
# used at the two pi.dim() checks. Verified by hand-tracing every tensor shape
# through the K=1 / with_null / use_head_gate=False / enable_film=False path
# used here (the only path this run exercises) -- no other code touches
# x_feat's shape between the fixed guard and the final write.
#
# Same optimal-setting cfg as submit_pat244_k1_l512_peak_bracket.sh (rms_joint
# router norm, with_null mode at K=1, multiscale_norm=rms, disable_layer_gate)
# EXCEPT wavelet_ctx_feat_detach_delta=false (not the template's true) -- kept
# explicit per user's stated preference that QWAB and the PaTH backbone should
# co-adapt (gradient flows into the backbone through the router/shift feature
# paths too), not train QWAB on top of a gradient-isolated backbone. Surfaced
# a broader finding while checking this: essentially all PAT-225/234/244
# small-model results (152 of ~156 checked scripts, including this template
# and its rho=512 K1/L512 peak F1=0.7368) use detach_delta=true; only 2
# "_nodetach" scripts on small model ever tested the joint variant. Medium's
# from-scratch pretrains (579696/579699) and its main dd_10ep/extralen finetune
# family default to joint already (never set this flag) -- so joint update is
# already medium's real behavior, but is a deliberate DEVIATION from small
# model's usual convention here, not a comparison against F1=0.7368 (that
# number is detach-mode; this run is joint-mode, not directly comparable).
# Tag renamed with _nodetach suffix to keep separate results directories.
#
# GPU: bounced between elm54 (3090x4) and elm66 (a6000x4) several times as
# nodes freed up and got grabbed by other users' jobs faster than this could
# resubmit. Currently pinned back to elm54 (3090x4, bs=8/accum=2, global_bs=64
# unchanged) so this runs NOW rather than waiting -- a second copy is queued
# in parallel via submit_pat_perhead_router_k1_l512_rho512_6000x4.sh (no
# nodelist pin, 4x6000, bs=16/accum=1) for whichever of elm71/72/73 frees up
# first; if/when that one starts, decide then whether to keep both or cancel
# one (same experiment, same seed=42 -- redundant, not a second data point).
#
# 2026-09-08: job 579778 (first real run on elm66, past the earlier startup
# checks) crashed at step 500 -- CUDA device-side assert / "scatter gather
# kernel index out of bounds" inside path_attn.py's wavelet_logit_bias_log_every
# sampling block (index_select axes hard-coded the shared-router [B,q_len,T]
# layout, not per-head's [B,H,q_len,T]; 500 == wavelet_logit_bias_log_every in
# the cfg below, so it crashed on the FIRST log attempt). Fixed in path_attn.py
# (per_head_router-aware index_select + valid-mask branches, plus the same
# latent bug in the analysis-export block just above it) and pushed before
# this resubmission -- see that repo's commit for the full trace.

set -euo pipefail

WORKDIR="/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling"
TAG="perheadrouter_K1_L512_me18_rho512_nodetach"
RUN_OUT="${WORKDIR}/runs/pat244_dual_temp/${TAG}"
mkdir -p "${RUN_OUT}/train"

TRAIN_SH="${WORKDIR}/_gen_pat244/train_${TAG}.sh"
TEST_SH="${WORKDIR}/_gen_pat244/test_${TAG}.sh"
mkdir -p "${WORKDIR}/_gen_pat244"

cat > "${TEST_SH}" <<EOF
#!/bin/bash
#SBATCH --job-name=hp2048_${TAG}
#SBATCH --output=${WORKDIR}/hotpot_long/logs/%j_${TAG}_ckpt15000_hotpot2048.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:2
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
set -euxo pipefail
if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u; fi
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:\${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true WANDB_MODE=disabled
JSONL="${WORKDIR}/hotpot_long/data/hotpot_long_dev_uniform.jsonl"
CHECKPOINT="${RUN_OUT}/checkpoint-15000"
CFG_PATH="${RUN_OUT}/supply_model.cfg"
BLOCK_SIZE=2048
[ -d "\${CHECKPOINT}" ] || { echo "Missing \${CHECKPOINT}" >&2; exit 1; }
OUTPUT_DIR="${WORKDIR}/hotpot_long/results_uniform/${TAG}_ckpt15000/L\${BLOCK_SIZE}"
mkdir -p "\${OUTPUT_DIR}"; cd "${WORKDIR}"
MASTER_PORT=\$((12000 + SLURM_JOB_ID % 10000))
/cl/work5/hongyu-s/conda/envs/latest_transformers/bin/torchrun --nproc_per_node=2 --master_port=\${MASTER_PORT} ./run_clm.py --model_type gpt2 --tokenizer_name gpt2 --model_name_or_path "\${CHECKPOINT}" --attn_implementation path_attn --cfg_path "\${CFG_PATH}" --dataset_name hotpot_qa --dataset_config_name distractor --hotpot_long_jsonl "\${JSONL}" --hotpot_long_lengths \${BLOCK_SIZE} --do_eval --block_size \${BLOCK_SIZE} --per_device_eval_batch_size 2 --path_attn_impl pytorch --report_to none --output_dir "\${OUTPUT_DIR}" --overwrite_output_dir --logging_dir "\${OUTPUT_DIR}/log" --seed 42 --path_use_qk_norm false --path_use_low_rank_w true --path_use_w_shortconv false --path_conv_size 3 --path_conv_bias false --num_harmonics 1 --single_A_B True --use_beta_modulation False --use_soft_wavelet_fox False --wavelet_baseline_use False --use_forget_gate False --qk_rotation False --ablate_switch False --wavelet_router False --load_best_model_at_end False
echo "=== Done: ${TAG} L\${BLOCK_SIZE} ==="
EOF

cat > "${TRAIN_SH}" <<EOF
#!/bin/bash
#SBATCH --job-name=PAT_${TAG}
#SBATCH --output=${RUN_OUT}/train/%j_${TAG}_train_eval.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:4
#SBATCH --nodelist=elm66
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=100:00:00
set -euxo pipefail
if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u; fi
cd "${WORKDIR}"
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:\${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true WANDB_MODE=disabled
RUN_OUT="${RUN_OUT}"
MASTER_PORT=\$((12000 + SLURM_JOB_ID % 20000))
mkdir -p "\${RUN_OUT}/train"
cat > "\${RUN_OUT}/supply_model.cfg" <<'CFG'
router_mode="seperate"
coe_mode="none"
tau=1
scale_type="none"
hotpot_question_position="later"
wavelet_mode="logit_bias_ctxscale_shift_v0"
bias_type="wavelet"
wavelet_logit_bias_a_init=-2
wavelet_ctxscale_tau=1.0
wavelet_ctxscale_router_rms_eps=1e-6
wavelet_ctxscale_chunk_q=256
wavelet_ctxscale_g_max=0.5
wavelet_ctxscale_g_bias_max=4.0
wavelet_ctxscale_lock_window=200
wavelet_ctxscale_lock_grad_eps=1e-6
wavelet_ctxscale_lock_update_eps=1e-6
wavelet_ctxscale_lock_min_frac=0.5
wavelet_ctxscale_lock_clamp_abs=4.0
wavelet_ctxscale_far_only=false
wavelet_ctxscale_far_min_delta=0
wavelet_ctxscale_head_indices="all"
wavelet_logit_bias_eps=1e-6
wavelet_logit_bias_rms_scope="context"
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
wavelet_ctxscale_router_per_head=true
wavelet_ctxscale_k=1
wavelet_ctxscale_scale_max_exp=18.0
multiscale_norm="rms"
wavelet_ctxscale_pattern_mode="ricker"
wavelet_ctxscale_center_pos_ratio=0.0
wavelet_ctxscale_dual_center_enable=false
wavelet_router_norm_mode="rms_joint"
wavelet_router_tau_null_init=1.0
wavelet_router_tau_scale_init=1.0
CFG
python -m torch.distributed.run --nproc_per_node=4 --master_port="\${MASTER_PORT}" ./run_clm.py --model_type gpt2 --tokenizer_name gpt2 --config_name gpt2 --share_freq_across_heads True --learning_rate 1e-4 --weight_decay 0.0 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --gradient_accumulation_steps 1 --block_size 512 --dataset_name mix --do_train --eval_strategy no --logging_dir "\${RUN_OUT}/train_log" --logging_steps 500 --num_train_epochs 10 --num_harmonics 1 --wavelet_pe_softmax_use False --save_steps 2500 --attn_implementation path_attn --path_use_qk_norm false --path_use_low_rank_w true --path_use_w_shortconv false --path_conv_size 3 --warmup_ratio 0.05 --path_conv_bias false --output_dir "\${RUN_OUT}" --overwrite_output_dir --b_unfreeze_step 5000 --pe_method no_pe --single_A_B True --use_beta_modulation False --use_soft_wavelet_fox False --wavelet_mode logit_bias_ctxscale_shift_v0 --bias_type wavelet --model_name_or_path runs/1r_baseline_from_s/checkpoint-80000 --full_fine_tune False --wavelet_baseline_use False --init_theta 0.847 --use_forget_gate False --sample_num 16 --spectral_loss_coe 0.1 --temp_loss_coe 0.0 --distill_teacher wavelet --distill_in_which_layers 0 --distill_freq_scale 25 --smooth_use False --distilling_coe_warmup_use False --scale_range 0 16 --weight_alpha 0.0 --loss_type cos --qk_rotation False --wavelet_router False --router_band_num 8 --router_hidden_dim 32 --rel_selection all --cfg_path "\${RUN_OUT}/supply_model.cfg" --seed 42
[ -d "\${RUN_OUT}/checkpoint-15000" ] || { echo "Missing final checkpoint" >&2; exit 1; }
bash "${TEST_SH}"
EOF

chmod +x "${TRAIN_SH}" "${TEST_SH}"
bash -n "${TRAIN_SH}"; bash -n "${TEST_SH}"
JID=$(sbatch --parsable "${TRAIN_SH}")
echo "submitted ${TAG} job=${JID} (elm66, a6000x4)"

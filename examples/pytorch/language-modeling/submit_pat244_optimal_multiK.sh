#!/bin/bash
# PAT-244: "optimal setting" (with_null_independent_scales + wavelet_router_norm_mode=rms_joint
# + multiscale_norm=rms, i.e. weight-then-sum-then-RMS-then-null-gate) extended to K3/K4/K5 at
# L_train in {512, 256}. No none-baseline pairing (router_rms delta already established at L256).
set -euo pipefail

WORKDIR="/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling"
RUN_BASE="${WORKDIR}/runs/pat244_dual_temp"
GEN_DIR="${WORKDIR}/_gen_pat244"
mkdir -p "${GEN_DIR}"

# args: tag  K  scale_max_exp_pycfg  block_size  chunk_q  nproc  bs  accum  gres_directive
emit_and_submit() {
  local tag="$1" K="$2" sme="$3" block_size="$4" chunk_q="$5" nproc="$6" bs="$7" accum="$8" gres="$9"
  local run_out="${RUN_BASE}/${tag}"
  local train_sh="${GEN_DIR}/train_${tag}.sh"
  local test_sh="${GEN_DIR}/test_${tag}.sh"
  mkdir -p "${run_out}/train"

  # ---- test (chained) script: eval at L2048, matches every other PAT-244 run ----
  cat > "${test_sh}" <<EOF
#!/bin/bash
#SBATCH --job-name=hp2048_${tag}
#SBATCH --output=${WORKDIR}/hotpot_long/logs/%j_${tag}_ckpt15000_hotpot2048.txt
#SBATCH --partition=gpu_long
${gres}
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
CHECKPOINT="${run_out}/checkpoint-15000"
CFG_PATH="${run_out}/supply_model.cfg"
BLOCK_SIZE=2048
[ -d "\${CHECKPOINT}" ] || { echo "Missing \${CHECKPOINT}" >&2; exit 1; }
OUTPUT_DIR="${WORKDIR}/hotpot_long/results_uniform/${tag}_ckpt15000/L\${BLOCK_SIZE}"
mkdir -p "\${OUTPUT_DIR}"; cd "${WORKDIR}"
MASTER_PORT=\$((12000 + SLURM_JOB_ID % 10000))
/cl/work5/hongyu-s/conda/envs/latest_transformers/bin/torchrun --nproc_per_node=${nproc} --master_port=\${MASTER_PORT} ./run_clm.py --model_type gpt2 --tokenizer_name gpt2 --model_name_or_path "\${CHECKPOINT}" --attn_implementation path_attn --cfg_path "\${CFG_PATH}" --dataset_name hotpot_qa --dataset_config_name distractor --hotpot_long_jsonl "\${JSONL}" --hotpot_long_lengths \${BLOCK_SIZE} --do_eval --block_size \${BLOCK_SIZE} --per_device_eval_batch_size 2 --path_attn_impl pytorch --report_to none --output_dir "\${OUTPUT_DIR}" --overwrite_output_dir --logging_dir "\${OUTPUT_DIR}/log" --seed 42 --path_use_qk_norm false --path_use_low_rank_w true --path_use_w_shortconv false --path_conv_size 3 --path_conv_bias false --num_harmonics 1 --single_A_B True --use_beta_modulation False --use_soft_wavelet_fox False --wavelet_baseline_use False --use_forget_gate False --qk_rotation False --ablate_switch False --wavelet_router False --load_best_model_at_end False
echo "=== Done: ${tag} L\${BLOCK_SIZE} ==="
EOF

  # ---- train script ----
  cat > "${train_sh}" <<EOF
#!/bin/bash
#SBATCH --job-name=PAT244_${tag}
#SBATCH --output=${run_out}/train/%j_${tag}_train_eval.txt
#SBATCH --partition=gpu_long
${gres}
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
RUN_OUT="${run_out}"
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
wavelet_ctxscale_chunk_q=${chunk_q}
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
wavelet_router_sigmoid_mode="with_null_independent_scales"
wavelet_ctx_feat_detach_delta=true
wavelet_ctx_feat_mode="q_minus_qcorr_meanh"
wavelet_ctxscale_k=${K}
wavelet_ctxscale_scale_max_exp=${sme}
multiscale_norm="rms"
wavelet_ctxscale_pattern_mode="ricker"
wavelet_ctxscale_center_pos_ratio=0.0
wavelet_ctxscale_dual_center_enable=false
wavelet_router_norm_mode="rms_joint"
CFG
python -m torch.distributed.run --nproc_per_node=${nproc} --master_port="\${MASTER_PORT}" ./run_clm.py --model_type gpt2 --tokenizer_name gpt2 --config_name gpt2 --share_freq_across_heads True --learning_rate 1e-4 --weight_decay 0.0 --per_device_train_batch_size ${bs} --per_device_eval_batch_size ${bs} --block_size ${block_size} --dataset_name mix --do_train --eval_strategy no --logging_dir "\${RUN_OUT}/train_log" --logging_steps 500 --num_train_epochs 10 --num_harmonics 1 --wavelet_pe_softmax_use False --save_steps 2500 --attn_implementation path_attn --path_use_qk_norm false --path_use_low_rank_w true --path_use_w_shortconv false --path_conv_size 3 --warmup_ratio 0.05 --path_conv_bias false --output_dir "\${RUN_OUT}" --overwrite_output_dir --gradient_accumulation_steps ${accum} --b_unfreeze_step 5000 --pe_method no_pe --single_A_B True --use_beta_modulation False --use_soft_wavelet_fox False --wavelet_mode logit_bias_ctxscale_shift_v0 --bias_type wavelet --model_name_or_path runs/1r_baseline_from_s/checkpoint-80000 --full_fine_tune False --wavelet_baseline_use False --init_theta 0.847 --use_forget_gate False --sample_num 16 --spectral_loss_coe 0.1 --temp_loss_coe 0.0 --distill_teacher wavelet --distill_in_which_layers 0 --distill_freq_scale 25 --smooth_use False --distilling_coe_warmup_use False --scale_range 0 16 --weight_alpha 0.0 --loss_type cos --qk_rotation False --wavelet_router False --router_band_num 8 --router_hidden_dim 32 --rel_selection all --cfg_path "\${RUN_OUT}/supply_model.cfg" --seed 42
[ -d "\${RUN_OUT}/checkpoint-15000" ] || { echo "Missing final checkpoint" >&2; exit 1; }
bash "${test_sh}"
EOF

  chmod +x "${train_sh}" "${test_sh}"
  bash -n "${train_sh}"; bash -n "${test_sh}"
  local jid
  jid=$(sbatch --parsable "${train_sh}")
  echo "submitted ${tag}  K=${K}  L=${block_size}  job=${jid}  (${gres})"
}

K4_SME='[8, 16, 20, 24]'
K5_SME='[14, 15.1998, 16, 16.6439, 17.1699]'

# --- L512 (native training length, chunk_q=128, nproc=2/bs=16/accum=2 -> global batch 64) ---
emit_and_submit K3_L512_optimal_center     3 '[22, 24, 25.169925001442312]' 512 128 2 16 2 '#SBATCH --gres=gpu:a100:2 --nodelist=elm43'
emit_and_submit K3_L512_optimal_k1combo    3 '[8, 16, 24]'                  512 128 2 16 2 '#SBATCH --gres=gpu:a100:2 --nodelist=elm43'
emit_and_submit K4_L512_optimal            4 "${K4_SME}"                    512 128 2 16 2 '#SBATCH --gres=gpu:6000:2 --nodelist=elm71'
emit_and_submit K5_L512_optimal            5 "${K5_SME}"                    512 128 2 16 2 '#SBATCH --gres=gpu:6000:2 --nodelist=elm71'

# --- L256 (chunk_q=256=T i.e. single chunk, nproc=2/bs=32/accum=1 -> global batch 64) ---
emit_and_submit K4_L256_optimal            4 "${K4_SME}"                    256 256 2 32 1 '#SBATCH --gres=gpu:6000:2 --nodelist=elm72'
emit_and_submit K5_L256_optimal            5 "${K5_SME}"                    256 256 2 32 1 '#SBATCH --gres=gpu:6000:2 --nodelist=elm73'

echo "=== all 6 optimal-config runs submitted ==="

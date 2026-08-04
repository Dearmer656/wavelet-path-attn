#!/bin/bash
# PAT-225 scale-cardinality ablation: fresh reruns (NOT resume) under the current
# "optimal setting" convention. All existing K>1 runs in pat225_scale_card were
# found to have a systemic bug: their supply_model.cfg never set
# wavelet_ctxscale_scale_max_exp for K>1, silently falling back to a scalar
# default (14.0) -- meaning every multi-scale arm (K=2,3,4,5,8,16) was actually
# training K IDENTICAL copies of the same scale, not a genuine multi-scale
# spread. This predates the 2026-07-28 validation commit that would have caught
# it (508a0c5eaa, fla/layers/path_attn.py) -- checkpoints for these runs were
# all created 2026-07-19/20, before that check existed.
#
# Router mode is K-dependent per the corrected PAT-244 "optimal setting" rule
# (see memory project_pat244_optimal_setting_router_mode.md): with_null at K=1,
# with_null_independent_scales at K>1 (with_null_independent_scales at K=1 stacks
# a redundant second sigmoid gate on g0_gate -- architecturally wrong).
set -euo pipefail

WORKDIR="/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling"
RUN_BASE="${WORKDIR}/runs/pat225_scale_card"
GEN_DIR="${WORKDIR}/_gen_pat225"
mkdir -p "${GEN_DIR}"

# args: tag  K  scale_max_exp_pycfg  seed  gres_directive
emit_and_submit() {
  local tag="$1" K="$2" sme="$3" seed="$4" gres="$5"
  local router_sigmoid_mode="with_null"
  if [ "${K}" -gt 1 ]; then
    router_sigmoid_mode="with_null_independent_scales"
  fi
  local run_out="${RUN_BASE}/${tag}"
  local train_sh="${GEN_DIR}/train_${tag}.sh"
  local test_sh="${GEN_DIR}/test_${tag}.sh"
  mkdir -p "${run_out}/train"

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
/cl/work5/hongyu-s/conda/envs/latest_transformers/bin/torchrun --nproc_per_node=2 --master_port=\${MASTER_PORT} ./run_clm.py --model_type gpt2 --tokenizer_name gpt2 --model_name_or_path "\${CHECKPOINT}" --attn_implementation path_attn --cfg_path "\${CFG_PATH}" --dataset_name hotpot_qa --dataset_config_name distractor --hotpot_long_jsonl "\${JSONL}" --hotpot_long_lengths \${BLOCK_SIZE} --do_eval --block_size \${BLOCK_SIZE} --per_device_eval_batch_size 2 --path_attn_impl pytorch --report_to none --output_dir "\${OUTPUT_DIR}" --overwrite_output_dir --logging_dir "\${OUTPUT_DIR}/log" --seed 42 --path_use_qk_norm false --path_use_low_rank_w true --path_use_w_shortconv false --path_conv_size 3 --path_conv_bias false --num_harmonics 1 --single_A_B True --use_beta_modulation False --use_soft_wavelet_fox False --wavelet_baseline_use False --use_forget_gate False --qk_rotation False --ablate_switch False --wavelet_router False --load_best_model_at_end False
echo "=== Done: ${tag} L\${BLOCK_SIZE} ==="
EOF

  cat > "${train_sh}" <<EOF
#!/bin/bash
#SBATCH --job-name=PAT225_${tag}_optrerun
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
wavelet_router_sigmoid_mode="${router_sigmoid_mode}"
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
python -m torch.distributed.run --nproc_per_node=2 --master_port="\${MASTER_PORT}" ./run_clm.py --model_type gpt2 --tokenizer_name gpt2 --config_name gpt2 --share_freq_across_heads True --learning_rate 1e-4 --weight_decay 0.0 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --block_size 512 --dataset_name mix --do_train --eval_strategy no --logging_dir "\${RUN_OUT}/train_log" --logging_steps 500 --num_train_epochs 10 --num_harmonics 1 --wavelet_pe_softmax_use False --save_steps 2500 --attn_implementation path_attn --path_use_qk_norm false --path_use_low_rank_w true --path_use_w_shortconv false --path_conv_size 3 --warmup_ratio 0.05 --path_conv_bias false --output_dir "\${RUN_OUT}" --overwrite_output_dir --gradient_accumulation_steps 2 --b_unfreeze_step 5000 --pe_method no_pe --single_A_B True --use_beta_modulation False --use_soft_wavelet_fox False --wavelet_mode logit_bias_ctxscale_shift_v0 --bias_type wavelet --model_name_or_path runs/1r_baseline_from_s/checkpoint-80000 --full_fine_tune False --wavelet_baseline_use False --init_theta 0.847 --use_forget_gate False --sample_num 16 --spectral_loss_coe 0.1 --temp_loss_coe 0.0 --distill_teacher wavelet --distill_in_which_layers 0 --distill_freq_scale 25 --smooth_use False --distilling_coe_warmup_use False --scale_range 0 16 --weight_alpha 0.0 --loss_type cos --qk_rotation False --wavelet_router False --router_band_num 8 --router_hidden_dim 32 --rel_selection all --cfg_path "\${RUN_OUT}/supply_model.cfg" --seed ${seed}
[ -d "\${RUN_OUT}/checkpoint-15000" ] || { echo "Missing final checkpoint" >&2; exit 1; }
bash "${test_sh}"
EOF

  chmod +x "${train_sh}" "${test_sh}"
  bash -n "${train_sh}"; bash -n "${test_sh}"
  local jid
  jid=$(sbatch --parsable "${train_sh}")
  echo "submitted ${tag}  K=${K}  seed=${seed}  job=${jid}  (${gres})"
}

K5_SME='[14, 15.1998, 16, 16.6439, 17.1699]'
K16_SME='[8, 9.0667, 10.1333, 11.2, 12.2667, 13.3333, 14.4, 15.4667, 16.5333, 17.6, 18.6667, 19.7333, 20.8, 21.8667, 22.9333, 24]'

emit_and_submit S1_s43_optrerun  1  14 43 '#SBATCH --gres=gpu:a100:2 --nodelist=elm43'
emit_and_submit S1_s44_optrerun  1  14 44 '#SBATCH --gres=gpu:a100:2 --nodelist=elm43'
emit_and_submit K5_s43_optrerun  5  "${K5_SME}" 43 '#SBATCH --gres=gpu:6000:2 --nodelist=elm71'
emit_and_submit K5_s44_optrerun  5  "${K5_SME}" 44 '#SBATCH --gres=gpu:6000:2 --nodelist=elm72'
emit_and_submit K16_s43_optrerun 16 "${K16_SME}" 43 '#SBATCH --gres=gpu:6000:2 --nodelist=elm73'

echo "=== all 5 submitted ==="

#!/bin/bash
#SBATCH --job-name=K3_L512_g0staticRmsBoth_128_256_384_s44
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat244_dual_temp/K3_L512_g0staticRmsBoth_128_256_384_s44/train/%j_K3_L512_g0staticRmsBoth_128_256_384_s44_train_eval.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:4
#SBATCH --nodelist=elm71
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=100:00:00
set -euxo pipefail
if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u; fi
cd "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling"
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true WANDB_MODE=disabled
RUN_OUT="/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat244_dual_temp/K3_L512_g0staticRmsBoth_128_256_384_s44"
MASTER_PORT=$((12000 + SLURM_JOB_ID % 20000))
mkdir -p "${RUN_OUT}/train"
cat > "${RUN_OUT}/supply_model.cfg" <<'CFG'
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
wavelet_router_sigmoid_mode="with_null_independent_scales"
wavelet_ctx_feat_detach_delta=true
wavelet_ctx_feat_mode="q_minus_qcorr_meanh"
wavelet_ctxscale_k=3
wavelet_ctxscale_scale_max_exp=[14, 16, 17.169925001442312]
wavelet_ctxscale_g0_learnable=true
multiscale_norm="rms_both"
wavelet_ctxscale_pattern_mode="ricker"
wavelet_ctxscale_center_pos_ratio=0.0
wavelet_ctxscale_dual_center_enable=false
wavelet_router_norm_mode="rms_joint"
wavelet_router_tau_null_init=1.0
wavelet_router_tau_scale_init=1.0
CFG
python -m torch.distributed.run --nproc_per_node=4 --master_port="${MASTER_PORT}" ./run_clm.py --model_type gpt2 --tokenizer_name gpt2 --config_name gpt2 --share_freq_across_heads True --learning_rate 1e-4 --weight_decay 0.0 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --block_size 512 --dataset_name mix --do_train --eval_strategy no --logging_dir "${RUN_OUT}/train_log" --logging_steps 500 --num_train_epochs 10 --num_harmonics 1 --wavelet_pe_softmax_use False --save_steps 2500 --attn_implementation path_attn --path_use_qk_norm false --path_use_low_rank_w true --path_use_w_shortconv false --path_conv_size 3 --warmup_ratio 0.05 --path_conv_bias false --output_dir "${RUN_OUT}" --overwrite_output_dir --gradient_accumulation_steps 1 --b_unfreeze_step 5000 --pe_method no_pe --single_A_B True --use_beta_modulation False --use_soft_wavelet_fox False --wavelet_mode logit_bias_ctxscale_shift_v0 --bias_type wavelet --model_name_or_path runs/1r_baseline_from_s/checkpoint-80000 --full_fine_tune False --wavelet_baseline_use False --init_theta 0.847 --use_forget_gate False --sample_num 16 --spectral_loss_coe 0.1 --temp_loss_coe 0.0 --distill_teacher wavelet --distill_in_which_layers 0 --distill_freq_scale 25 --smooth_use False --distilling_coe_warmup_use False --scale_range 0 16 --weight_alpha 0.0 --loss_type cos --qk_rotation False --wavelet_router False --router_band_num 8 --router_hidden_dim 32 --rel_selection all --cfg_path "${RUN_OUT}/supply_model.cfg" --seed 44
[ -d "${RUN_OUT}/checkpoint-15000" ] || { echo "Missing final checkpoint" >&2; exit 1; }
bash "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_k3_g0static_rmsboth/test_K3_L512_g0staticRmsBoth_128_256_384_s44.sh"

#!/bin/bash
#SBATCH --job-name=QWABExtraLen2048
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/log_file/train/%j_qwab_medium_dd_extralen2048_s42_fp32.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:2
#SBATCH --nodelist=elm71
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# PAT-226: QWAB medium (mix_medium_owt_dd_10ep/checkpoint-15000, the "QWAB headline"
# medium config, s42 -- logit_bias_ctxscale_shift_v0, path_attn, distill_teacher=wavelet)
# given the SAME target-length adaptation budget as the Rotary+YaRN comparison
# (train_gpt2_medium_owt_mix_rotary_yarn_2048_s42_fp32.sh): 400 steps at block_size=2048,
# lr=2e-5, warmup_ratio=0.05, global_bs=64 -- deliberately copying YaRN's finetune recipe
# exactly (not QWAB's own 1e-4 finetune-stage convention) so this is a clean "give QWAB
# the same extra adaptation opportunity YaRN needed" comparison, not a new confound.
# Standard --do_train (no freeze flags) updates ALL parameters, per explicit request --
# --b_unfreeze_step is carried over from the original QWAB finetune script but is a
# documented NO-OP in the current run_clm.py (the B-branch freeze mechanism was removed
# from trainer.py), so it has no effect either way.
# Purpose: QWAB's own zero-shot L2048 (0.8191, from the 512-length finetune, no extra
# adaptation) already beats YaRN's post-finetune L2048 (0.6031) -- this run checks whether
# giving QWAB the same extra 400-step L2048 budget changes that picture (pre-empting the
# "you didn't give QWAB equal adaptation opportunity" question).
# Batch size matches the Rotary+YaRN L2048 finetune (1/accum, not QWAB's own 4/accum4)
# since block_size=2048 (4x the original 512) needs the same memory headroom reduction.
# 2026-09-07: switched 4xa6000(accum16) -> 2xa6000(accum32) per request, global_bs=64
# unchanged (1*2*32=1*4*16=64). OOM'd on BOTH 4x3090 (24GB) and 2xa6000 (48GB) at
# per_device_train_batch_size=1 -- distill_teacher=wavelet's extra teacher forward pass at
# block_size=2048 needs >48GB even at the minimum batch, well beyond what plain Rotary+YaRN
# (no distillation branch) required at the same batch/length. Added
# --gradient_checkpointing True to cut activation memory (trades recompute for memory).
# That alone then hit "Expected to have finished reduction in the prior iteration" --
# gradient checkpointing recomputes forward during backward, which combined with QWAB's
# conditionally-used wavelet/distillation branches confuses DDP's gradient-ready bucketing;
# added --ddp_find_unused_parameters True per the error's own suggestion. That then hit
# "Expected to mark a variable ready only once" (reused params across reentrant backward
# passes); switched to --gradient_checkpointing_kwargs '{"use_reentrant": false}'.
# 2026-09-07 (later): verified --gradient_checkpointing is actually a NO-OP in this fork's
# GPT2Model -- grepped modeling_gpt2.py, there is no torch.utils.checkpoint call anywhere;
# `self.gradient_checkpointing` only gates whether past_key_values gets passed, unrelated to
# activation recomputation. So none of the three checkpointing-related flags above were
# doing anything (the run that succeeded, 578411, did so for some other/unconfirmed reason,
# not because of them). Reverted to the clean recipe (no checkpointing flags) and switched
# to 2xp6000 (24GB each) per explicit request.

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
cd "${WORKDIR}"
PRETRAIN_CKPT="${WORKDIR}/runs/mix_medium_owt_dd_10ep/checkpoint-15000"
# OUT dir name kept as "dd_yarn2048" (not renamed to match this script's new filename)
# so this stays pointed at the same output as the already-running job (578366) --
# renaming it here would orphan that job's in-progress checkpoint into an unreferenced dir.
OUT="${WORKDIR}/runs/mix_medium_owt_dd_yarn2048_s42_fp32"
mkdir -p "${OUT}"

# 2026-09-07: job 578366 FAILED (3min in) with "wavelet_ctxscale_scale_max_exp must be a
# list/tuple of length 8 when wavelet_ctxscale_k=8, not a single value: 14.0". This
# checkpoint predates the wavelet_ctxscale_k/scale_max_exp fields entirely (absent from
# both its own supply_model.cfg and its saved config.json) -- the current codebase's
# defaults (k=8, scale_max_exp=14.0 scalar) are mutually incompatible with each other's own
# validation, a latent bug unrelated to this script. Writing an explicit list here to
# satisfy the check, matching the original checkpoint's supply_model.cfg content plus this
# one addition (rather than fixing the codebase default, out of scope here).
cat "${WORKDIR}/runs/mix_medium_owt_dd_10ep/supply_model.cfg" > "${OUT}/supply_model.cfg"
echo "wavelet_ctxscale_scale_max_exp=[14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0]" >> "${OUT}/supply_model.cfg"

MASTER_PORT=$(( 24800 + SLURM_JOB_ID % 1000 ))

echo "=== QWAB medium (dd headline) extra L2048 finetune (400 steps, matching YaRN's recipe): 4xa6000 ==="

/cl/work5/hongyu-s/conda/envs/latest_transformers/bin/torchrun \
  --nproc_per_node=2 \
  --master_port="${MASTER_PORT}" \
  ./run_clm.py \
  --model_type gpt2 \
  --tokenizer_name gpt2 \
  --model_name_or_path "${PRETRAIN_CKPT}" \
  --dataset_name mix \
  --block_size 2048 \
  --do_train \
  --max_steps 400 \
  --logging_steps 20 \
  --save_steps 400 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --learning_rate 2e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.05 \
  --adam_beta2 0.95 \
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
  --preprocessing_num_workers 8 \
  --ddp_timeout 21600 \
  --seed 42 \
  --overwrite_output_dir \
  --output_dir "${OUT}" \
  --logging_dir "${OUT}/tensorboard" \
  --cfg_path "${OUT}/supply_model.cfg"

echo "=== QWAB medium (dd headline) extra L2048 finetune done ==="

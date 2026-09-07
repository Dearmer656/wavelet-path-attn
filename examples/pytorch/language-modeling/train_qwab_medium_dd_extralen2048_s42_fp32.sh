#!/bin/bash
#SBATCH --job-name=QWABExtraLen2048
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/log_file/train/%j_qwab_medium_dd_extralen2048_s42_fp32.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:4
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
# Batch size matches the Rotary+YaRN L2048 finetune (1/accum16, not QWAB's own 4/accum4)
# since block_size=2048 (4x the original 512) needs the same memory headroom reduction.

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

MASTER_PORT=$(( 24800 + SLURM_JOB_ID % 1000 ))

echo "=== QWAB medium (dd headline) extra L2048 finetune (400 steps, matching YaRN's recipe): 4xa6000 ==="

/cl/work5/hongyu-s/conda/envs/latest_transformers/bin/torchrun \
  --nproc_per_node=4 \
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
  --gradient_accumulation_steps 16 \
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

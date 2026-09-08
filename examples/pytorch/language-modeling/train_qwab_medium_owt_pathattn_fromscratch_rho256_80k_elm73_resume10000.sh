#!/bin/bash
#SBATCH --job-name=qwab_med_scratch_rho256_elm73
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/log_file/train/%j_qwab_medium_owt_pathattn_fromscratch_rho256_80k_elm73_resume10000.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:4
#SBATCH --nodelist=elm73
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# PAT-226: move the rho=256 QWAB medium from-scratch pretrain (job 579699, running on elm71)
# to elm73 (fastest node, only pinned because it's the explicit target here) by resuming
# from checkpoint-10000 -- the most recent saved checkpoint on disk (save_steps=10000;
# progress was 14572/80000 on elm71 when this was submitted, past 10000 but before the next
# save point at 20000, so checkpoint-10000 is the latest available snapshot).
# 579699 keeps running untouched on elm71 in parallel (same reasoning as the earlier
# per-head-router node races this session: whichever finishes/produces results first).
# --resume_from_checkpoint restores optimizer/scheduler/step state via HF Trainer (verified
# in run_clm.py: training_args.resume_from_checkpoint is passed straight to
# trainer.train(resume_from_checkpoint=...), independent of --overwrite_output_dir, which
# only gates an earlier auto-detect/error-check block and does not delete anything).
# IMPORTANT: this writes to a SEPARATE output_dir (..._elm73resume10000, not 579699's own
# runs/gpt2_medium_owt_qwab_pathattn_fromscratch_rho256_80k) -- reads the resume checkpoint
# FROM 579699's directory but never writes back into it, so the two jobs cannot race-corrupt
# each other's checkpoint files while both are running. Decide which copy to keep once one
# gets meaningfully ahead.

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
SRC_OUT="${WORKDIR}/runs/gpt2_medium_owt_qwab_pathattn_fromscratch_rho256_80k"
RESUME_CKPT="${SRC_OUT}/checkpoint-10000"
[ -d "${RESUME_CKPT}" ] || { echo "Missing ${RESUME_CKPT}" >&2; exit 1; }
OUT="${WORKDIR}/runs/gpt2_medium_owt_qwab_pathattn_fromscratch_rho256_80k_elm73resume10000"
mkdir -p "${OUT}/train"

CFG_PATH="${OUT}/supply_model.cfg"
cat > "${CFG_PATH}" <<'CFG'
wavelet_ctxscale_k=1
wavelet_ctxscale_scale_max_exp=16.0
CFG

MASTER_PORT=$(( 24950 + SLURM_JOB_ID % 1000 ))

echo "=== QWAB medium FROM-SCRATCH OWT pretrain, rho=256: 4x6000 (elm73, resume from checkpoint-10000) ==="

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
  --resume_from_checkpoint "${RESUME_CKPT}" \
  --output_dir "${OUT}" \
  --logging_dir "${OUT}/train/tensorboard" \
  --cfg_path "${CFG_PATH}"

echo "=== QWAB medium from-scratch OWT pretrain (rho=256, elm73 resume) done ==="

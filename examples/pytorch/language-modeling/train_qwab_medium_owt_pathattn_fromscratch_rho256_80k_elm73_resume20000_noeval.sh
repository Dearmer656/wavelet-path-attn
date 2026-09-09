#!/bin/bash
#SBATCH --job-name=qwab_med_scratch_rho256_elm73_noeval
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/log_file/train/%j_qwab_medium_owt_pathattn_fromscratch_rho256_80k_elm73_resume20000_noeval.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:4
#SBATCH --nodelist=elm73
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# PAT-226: resume the elm73 rho=256 QWAB medium pretrain from checkpoint-20000, with eval
# DISABLED. The previous job (580784) was found to run a full eval every 5000 steps taking
# ~1640s (~27min) each (eval_runtime confirmed directly in its log at steps 15000/20000/25000/
# 30000) -- a large, avoidable overhead for a from-scratch pretrain that doesn't need
# periodic validation-loss tracking. checkpoint-30000 was accidentally deleted by the user
# before this was written, so this resumes from checkpoint-20000 (the latest surviving one,
# in this SAME job's own output dir -- this is now a genuine in-place resume, not a
# cross-directory read like the earlier ..._resume10000 script, since checkpoint-20000 lives
# in this run's own output_dir already).
# Per project convention (memory: "disable eval, keep stats" -- future training should drop
# --do_eval to save time but keep training-time wavelet stats, which are unaffected by this
# flag): --eval_strategy no, --do_eval removed. --load_best_model_at_end/--metric_for_best_model
# /--greater_is_better also removed since load_best_model_at_end requires eval_strategy != no
# in HF Trainer and is meaningless without periodic eval anyway -- the FINAL checkpoint at
# max_steps=80000 is what will be used, not a "best" one.

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
OUT="${WORKDIR}/runs/gpt2_medium_owt_qwab_pathattn_fromscratch_rho256_80k_elm73resume10000"
RESUME_CKPT="${OUT}/checkpoint-20000"
[ -d "${RESUME_CKPT}" ] || { echo "Missing ${RESUME_CKPT}" >&2; exit 1; }
mkdir -p "${OUT}/train"

CFG_PATH="${OUT}/supply_model.cfg"
cat > "${CFG_PATH}" <<'CFG'
wavelet_ctxscale_k=1
wavelet_ctxscale_scale_max_exp=16.0
CFG

MASTER_PORT=$(( 24960 + SLURM_JOB_ID % 1000 ))

echo "=== QWAB medium FROM-SCRATCH OWT pretrain, rho=256: 4x6000 (elm73, resume from checkpoint-20000, eval disabled) ==="

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
  --do_train \
  --max_steps 80000 \
  --eval_strategy no \
  --save_steps 10000 \
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

echo "=== QWAB medium from-scratch OWT pretrain (rho=256, elm73 resume, no-eval) done ==="

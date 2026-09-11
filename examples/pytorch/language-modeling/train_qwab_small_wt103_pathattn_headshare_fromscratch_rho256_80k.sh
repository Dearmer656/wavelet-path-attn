#!/bin/bash
#SBATCH --job-name=qwab_small_headshare_rho256
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/log_file/train/%j_qwab_small_wt103_pathattn_headshare_fromscratch_rho256_80k.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:4
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# small-model counterpart to the medium QWAB rho256 from-scratch pretrain
# (job 581523, runs/gpt2_medium_owt_qwab_pathattn_fromscratch_rho256_80k_elm73resume10000),
# to extend the PAT-258 "does pretraining WITH QWAB active matter" question
# to small scale too. GPT-2 small (12 layers), WikiText-103 (matches this
# project's own documented small-model Stage-1 pretrain corpus, NOT
# OpenWebText which medium uses), HEAD-SHARED QWAB router (the standard/
# production router mode -- explicitly NOT wavelet_ctxscale_router_per_head,
# which is a separate experimental line elsewhere in this project), rho=256
# (wavelet_ctxscale_scale_max_exp=16.0) to match medium for later
# small-vs-medium comparability, per explicit user confirmation.
# Hyperparameters mirror REPRODUCIBILITY_APPENDIX_NOTES.md's documented
# small-model Stage-1 recipe (lr=1e-4, warmup=5000 absolute steps, global
# batch 64, 80000 steps) and this session's medium from-scratch scripts'
# CLI shape (bs=8 x accum=2 x 4 GPUs = 64 global).

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
OUT="${WORKDIR}/runs/gpt2_small_wt103_qwab_pathattn_fromscratch_rho256_headshare_80k"
mkdir -p "${OUT}/train"

CFG_PATH="${OUT}/supply_model.cfg"
cat > "${CFG_PATH}" <<'CFG'
wavelet_ctxscale_k=1
wavelet_ctxscale_scale_max_exp=16.0
CFG

echo "=== supply_model.cfg written to ${CFG_PATH}: ==="
cat "${CFG_PATH}"
echo "=== end cfg ==="

MASTER_PORT=$(( 25970 + SLURM_JOB_ID % 1000 ))

echo "=== QWAB SMALL from-scratch WikiText-103 pretrain, rho=256, head-share: 4x6000 ==="

/cl/work5/hongyu-s/conda/envs/latest_transformers/bin/torchrun \
  --nproc_per_node=4 \
  --master_port="${MASTER_PORT}" \
  ./run_clm.py \
  --model_type gpt2 \
  --tokenizer_name gpt2 \
  --config_name openai-community/gpt2 \
  --dataset_name wikitext \
  --dataset_config_name wikitext-103-raw-v1 \
  --block_size 512 \
  --do_train \
  --max_steps 80000 \
  --eval_strategy no \
  --save_steps 10000 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-4 \
  --weight_decay 0.0 \
  --warmup_steps 5000 \
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

echo "=== QWAB small from-scratch WikiText-103 pretrain (rho=256, head-share) done ==="

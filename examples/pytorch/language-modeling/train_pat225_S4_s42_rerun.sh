#! /bin/bash
#SBATCH --job-name=PAT225_S4_s42_rerun
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat225_scale_card/S4_s42_rerun/train/%j_pat225_S4_s42_rerun.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:2
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# PAT-225 reproducibility check: re-run S4_s42 (K=4, seed=42) with the exact
# same recipe/seed to test whether distributed-training non-determinism (no
# cudnn.deterministic / use_deterministic_algorithms set anywhere in the
# pipeline) means "seed=42" doesn't actually reproduce the original 0.7038
# L4096 F1 result. 2x A6000 instead of 4x; per-device batch stays 16 (matching
# the original), gradient_accumulation_steps=2 to preserve global batch=64.
# Output goes to S4_s42_rerun/ — does NOT overwrite the original S4_s42 checkpoint.
# Branch: hongyusaatitech/pat-225-sensei-ablation-is-qwab-genuinely-multi-scale-scale

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi

WORKDIR=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
cd "${WORKDIR}"

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true
export WANDB_MODE=disabled

RUN_OUT="${WORKDIR}/runs/pat225_scale_card/S4_s42_rerun"
MASTER_PORT=$(( 24225 + SLURM_JOB_ID % 1000 ))

echo "================= BEGIN RUN PAT-225 S=4 s42 RERUN (reproducibility check) ================="

python -m torch.distributed.run --nproc_per_node=4 --master_port=${MASTER_PORT} ./run_clm.py \
  --model_type gpt2 --tokenizer_name gpt2 --config_name gpt2 \
  --share_freq_across_heads True \
  --learning_rate 1e-4 --weight_decay 0.0 \
  --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
  --block_size 512 --dataset_name mix \
  --do_train --do_eval --eval_strategy steps --eval_steps 500 \
  --logging_dir "${RUN_OUT}/train_log" --logging_steps 500 \
  --num_train_epochs 10 --num_harmonics 1 \
  --wavelet_pe_softmax_use False \
  --save_steps 2500 \
  --attn_implementation path_attn \
  --path_use_qk_norm false --path_use_low_rank_w true \
  --path_use_w_shortconv false --path_conv_size 3 \
  --warmup_ratio 0.05 --path_conv_bias false \
  --output_dir "${RUN_OUT}" --overwrite_output_dir \
  --gradient_accumulation_steps 1 \
  --b_unfreeze_step 5000 --pe_method no_pe --single_A_B True \
  --use_beta_modulation False --use_soft_wavelet_fox False \
  --wavelet_mode logit_bias_ctxscale_shift_v0 \
  --model_name_or_path runs/1r_baseline_from_s/checkpoint-80000 \
  --full_fine_tune False \
  --wavelet_baseline_use False --init_theta 0.847 \
  --use_forget_gate False --sample_num 16 \
  --spectral_loss_coe 0.1 --temp_loss_coe 0.0 \
  --distill_teacher wavelet --distill_in_which_layers 0 \
  --distill_freq_scale 25 --smooth_use False \
  --distilling_coe_warmup_use False --scale_range 0 16 \
  --weight_alpha 0.0 --loss_type cos --qk_rotation False \
  --wavelet_router False \
  --router_band_num 8 --router_hidden_dim 32 --rel_selection all \
  --cfg_path "${RUN_OUT}/supply_model.cfg" \
  --seed 42

echo "=== PAT-225 S=4 s42 rerun done ==="

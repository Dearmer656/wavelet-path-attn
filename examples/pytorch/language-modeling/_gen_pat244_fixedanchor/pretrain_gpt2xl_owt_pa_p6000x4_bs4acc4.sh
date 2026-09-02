#!/bin/bash
#SBATCH --job-name=pretrain_gpt2xl_owt_pa_p6000x4
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/gpt2xl_owt_pa_p6000x4/%j_pretrain.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:p6000:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=100:00:00
set -euxo pipefail
if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u; fi
cd "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling"
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN_OUT="/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/gpt2xl_owt_pa_p6000x4"
mkdir -p "${RUN_OUT}"
MASTER_PORT=$((12000 + SLURM_JOB_ID % 10000))
cat > "${RUN_OUT}/supply_model.cfg" <<'CFG'
router_mode="seperate"
coe_mode="none"
tau=1
scale_type="none"
wavelet_ctxscale_k=1
CFG
torchrun --nproc_per_node=4 --master_port="${MASTER_PORT}" ./run_clm.py \
  --model_type gpt2 \
  --tokenizer_name gpt2 \
  --config_name openai-community/gpt2-xl \
  --dataset_name openwebtext \
  --validation_split_percentage 1 \
  --block_size 512 \
  --do_train \
  --do_eval \
  --max_steps 80000 \
  --skip_memory_metrics False \
  --eval_strategy steps \
  --eval_steps 5000 \
  --save_steps 20000 \
  --load_best_model_at_end True \
  --metric_for_best_model eval_loss \
  --greater_is_better False \
  --logging_steps 500 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --warmup_ratio 0.05 \
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
  --wavelet_pe_softmax_use True \
  --wavelet_mode db1 \
  --wavelet_baseline_use False \
  --wavelet_router False \
  --use_beta_modulation False \
  --use_soft_wavelet_fox False \
  --use_forget_gate False \
  --full_fine_tune False \
  --init_theta 0.847 \
  --sample_num 16 \
  --spectral_loss_coe 0.0 \
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
  --seed 42 \
  --overwrite_output_dir \
  --output_dir "${RUN_OUT}" \
  --logging_dir "${RUN_OUT}/log" \
  --cfg_path "${RUN_OUT}/supply_model.cfg"
echo "=== gpt2-xl PA-only pretrain p6000x4 bs4acc4 (global_bs=64) done ==="

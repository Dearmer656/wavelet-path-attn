#!/bin/bash
#SBATCH --job-name=MedMix_PAonly_s43
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/mix_medium_PA_only_s43/train/%j_train.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:p6000:4
#SBATCH --nodelist=elm82
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
set -euxo pipefail
if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u; fi
cd /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true WANDB_MODE=disabled
RUN_OUT="/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/mix_medium_PA_only_s43"
mkdir -p "${RUN_OUT}/train"
MASTER_PORT=$(( 23442 + SLURM_JOB_ID % 1000 ))
cat > "${RUN_OUT}/supply_model.cfg" <<'CFG'
router_mode="seperate"
coe_mode="none"
tau=1
scale_type="none"
hotpot_question_position="later"
wavelet_ctxscale_k=1
CFG
/cl/work5/hongyu-s/conda/envs/latest_transformers/bin/torchrun \
  --nproc_per_node=4 \
  --master_port="${MASTER_PORT}" \
  ./run_clm.py \
  --model_type gpt2 \
  --tokenizer_name gpt2 \
  --config_name openai-community/gpt2-medium \
  --model_name_or_path runs/gpt2_medium_owt_pytorch_level_path_attn/checkpoint-80000 \
  --dataset_name mix \
  --block_size 512 \
  --do_train \
  --num_train_epochs 10 \
  --logging_steps 500 \
  --save_steps 5000 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --weight_decay 0.0 \
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
  --wavelet_pe_softmax_use False \
  --wavelet_mode db1 \
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
  --seed 43 \
  --overwrite_output_dir \
  --output_dir "${RUN_OUT}" \
  --logging_dir "${RUN_OUT}/train_log" \
  --cfg_path "${RUN_OUT}/supply_model.cfg"
[ -d "${RUN_OUT}/checkpoint-15000" ] || { echo "Missing final checkpoint" >&2; exit 1; }
sbatch "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_pat244_medium/test_mix_medium_PA_only_s43_L512_2048.sh"
sbatch "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_pat244_medium/test_mix_medium_PA_only_s43_L4096_p6000.sh"
sbatch "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_pat244_medium/test_mix_medium_PA_only_s43_L8192_triton.sh"
sbatch "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_pat244_medium/test_mix_medium_PA_only_s43_L12288_triton.sh"
sbatch "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_pat244_medium/test_mix_medium_PA_only_s43_L16384_triton.sh"
sbatch "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_xsum_basis/run_medium_PAonly_s43_xsum.sh"
echo "=== Done: mix_medium_PA_only_s43 ==="

#!/bin/bash
#SBATCH --job-name=postfix_PA_only_s42
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat225_postfix_docorder/PA_only_s42/train/%j_PA_only_s42_train_eval.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:4
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
RUN_OUT="/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat225_postfix_docorder/PA_only_s42"
MASTER_PORT=$((12000 + SLURM_JOB_ID % 20000))
mkdir -p "${RUN_OUT}/train"
cat > "${RUN_OUT}/supply_model.cfg" <<'CFG'
router_mode="seperate"
coe_mode="none"
wavelet_mode="off"
wavelet_ctxscale_k=1
wavelet_ctxscale_scale_max_exp=14
CFG
python -m torch.distributed.run --nproc_per_node=4 --master_port="${MASTER_PORT}" ./run_clm.py --model_type gpt2 --tokenizer_name gpt2 --config_name gpt2 --share_freq_across_heads True --learning_rate 1e-4 --weight_decay 0.0 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --block_size 512 --dataset_name mix --do_train --eval_strategy no --logging_dir "${RUN_OUT}/train_log" --logging_steps 500 --num_train_epochs 10 --num_harmonics 1 --wavelet_pe_softmax_use False --save_steps 2500 --attn_implementation path_attn --path_use_qk_norm false --path_use_low_rank_w true --path_use_w_shortconv false --path_conv_size 3 --warmup_ratio 0.05 --path_conv_bias false --output_dir "${RUN_OUT}" --overwrite_output_dir --gradient_accumulation_steps 1 --pe_method no_pe --single_A_B True --use_beta_modulation False --use_soft_wavelet_fox False --wavelet_mode off --model_name_or_path runs/1r_baseline_from_s/checkpoint-80000 --full_fine_tune False --wavelet_baseline_use False --init_theta 0.847 --use_forget_gate False --qk_rotation False --wavelet_router False --distill_in_which_layers 0 --spectral_loss_coe 0.0 --temp_loss_coe 0.0 --cfg_path "${RUN_OUT}/supply_model.cfg" --seed 42
[ -d "${RUN_OUT}/checkpoint-15000" ] || { echo "Missing final checkpoint" >&2; exit 1; }
bash "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_postfix_4way_s42/eval_PA_only_s42.sh"

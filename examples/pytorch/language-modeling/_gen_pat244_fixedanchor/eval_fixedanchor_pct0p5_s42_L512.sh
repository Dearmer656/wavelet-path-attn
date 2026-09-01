#!/bin/bash
#SBATCH --job-name=eval_fixedanchor_pct0p5
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat244_dual_temp/K1_L512_rho256_fixedanchor_pct0p5_s42/eval/%j_eval.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
set -euxo pipefail
if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u; fi
cd "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling"
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/cl/work5/hongyu-s/worktrees/qwab-fixed-abs-anchor:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true WANDB_MODE=disabled
CHECKPOINT="/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat244_dual_temp/K1_L512_rho256_fixedanchor_pct0p5_s42/checkpoint-15900"
CFG_PATH="/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat244_dual_temp/K1_L512_rho256_fixedanchor_pct0p5_s42/supply_model.cfg"
OUTPUT_DIR="/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat244_dual_temp/K1_L512_rho256_fixedanchor_pct0p5_s42/eval_L512_target0p5"
mkdir -p "${OUTPUT_DIR}"
MASTER_PORT=$((12000 + SLURM_JOB_ID % 10000))
torchrun --nproc_per_node=1 --master_port=${MASTER_PORT} ./run_clm.py --model_type gpt2 --tokenizer_name gpt2 --model_name_or_path "${CHECKPOINT}" --attn_implementation path_attn --cfg_path "${CFG_PATH}" --dataset_name hotpot_qa --dataset_config_name distractor --do_eval --block_size 512 --per_device_eval_batch_size 8 --path_attn_impl pytorch --report_to none --output_dir "${OUTPUT_DIR}" --overwrite_output_dir --logging_dir "${OUTPUT_DIR}/log" --seed 42 --path_use_qk_norm false --path_use_low_rank_w true --path_use_w_shortconv false --path_conv_size 3 --path_conv_bias false --num_harmonics 1 --single_A_B True --use_beta_modulation False --use_soft_wavelet_fox False --wavelet_baseline_use False --use_forget_gate False --qk_rotation False --ablate_switch False --wavelet_router False --load_best_model_at_end False
echo "=== Done: fixedanchor pct0.5 eval @ target0.5 ==="

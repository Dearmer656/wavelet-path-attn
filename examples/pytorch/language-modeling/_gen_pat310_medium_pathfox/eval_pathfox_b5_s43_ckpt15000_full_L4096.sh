#!/bin/bash
#SBATCH --job-name=PAT310_pathfox_b5_ck15000_full_L4096
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/results_uniform/pat310_medium_pathfox/logs/%j_pathfox_b5_s43_ckpt15000_full_L4096.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:p6000:1
#SBATCH --exclude=elm72,elm26
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# PaTH-FoX b5 s43 checkpoint-15000, full paper-protocol eval at L4096 (triton; the forget gate is only applied by the triton kernel).
set -euxo pipefail
if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u; fi
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets WANDB_DISABLED=true WANDB_MODE=disabled
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID:-manual}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
RUN=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat310_medium_pathfox/pathfox_b5_s43
CK=$RUN/checkpoint-15000
OUT=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/results_uniform/pat310_medium_pathfox/pathfox_b5_s43/ckpt15000_full/L4096; mkdir -p $OUT
python ./run_clm.py --model_type gpt2 --tokenizer_name gpt2 --model_name_or_path $CK --cfg_path $RUN/supply_model.cfg --dataset_name hotpot_qa --dataset_config_name distractor --hotpot_long_jsonl /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/hotpot_long_dev_uniform.jsonl --hotpot_long_lengths 4096 --do_eval --block_size 4096 --per_device_eval_batch_size 1 --path_attn_impl triton --report_to none --output_dir $OUT --overwrite_output_dir --logging_dir $OUT/log --seed 42 --attn_implementation path_attn --path_use_qk_norm false --path_use_low_rank_w true --path_use_w_shortconv false --path_conv_size 3 --path_conv_bias false --single_A_B True --use_beta_modulation False --use_soft_wavelet_fox False --wavelet_baseline_use False --use_forget_gate True --qk_rotation False --wavelet_router False --path_conv_bias false --num_harmonics 1 --ablate_switch False --load_best_model_at_end False

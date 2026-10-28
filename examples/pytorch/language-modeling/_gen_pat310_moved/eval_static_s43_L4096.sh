#!/bin/bash
#SBATCH --job-name=PAT310_moved_static_s43_L4096
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/results_uniform/pat310_moved/static_s43/L4096/%j_eval.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:p6000:2
#SBATCH --exclude=elm72
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=100:00:00
# PAT-310: HotpotQA-Long with evidence at its uniform-random position (hotpot_long_dev_uniform.jsonl + hotpot_respect_doc_order=True via sidecar cfg), instead of front-pinned.
set -euxo pipefail
set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets WANDB_DISABLED=true WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
MASTER_PORT=$((12000 + SLURM_JOB_ID % 10000))
torchrun --nproc_per_node=2 --master_port=${MASTER_PORT} ./run_clm.py --model_type gpt2 --tokenizer_name gpt2 --model_name_or_path /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat310_static/K1_L512_me16_rho256_ricker_static_s43/checkpoint-15000 --attn_implementation path_attn --cfg_path /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_pat310_moved/cfgs/static_s43.cfg --dataset_name hotpot_qa --dataset_config_name distractor --hotpot_long_jsonl /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/hotpot_long_dev_uniform.jsonl --hotpot_long_lengths 4096 --do_eval --block_size 4096 --per_device_eval_batch_size 2 --path_attn_impl pytorch --use_forget_gate False --bias_type wavelet --report_to none --output_dir /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/results_uniform/pat310_moved/static_s43/L4096 --overwrite_output_dir --logging_dir /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/results_uniform/pat310_moved/static_s43/L4096/log --seed 42 --path_use_qk_norm false --path_use_low_rank_w true --path_use_w_shortconv false --path_conv_size 3 --path_conv_bias false --num_harmonics 1 --single_A_B True --use_beta_modulation False --use_soft_wavelet_fox False --wavelet_baseline_use False --qk_rotation False --ablate_switch False --wavelet_router False --load_best_model_at_end False

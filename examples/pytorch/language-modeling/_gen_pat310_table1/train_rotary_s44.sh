#!/bin/bash
#SBATCH --job-name=T1_train_rotary_s44
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/log_file/train/%j_mix_medium_owt_rotary_10ep_s44_fp32.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:p6000:2
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# PAT-310 Table 1: rotary medium mix fine-tune, seed 44. cfg = the s42 run's saved training_args (per-device bs 2, attn eager,
# lr 1e-4, wd 0, warmup_ratio 0.05, 10 epochs, linear, fp32); 2 GPUs x accum 16 = global batch 64 (s42 used 4 GPUs x accum 8).
set -euxo pipefail
set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets WANDB_DISABLED=true WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SKIP_FENICE=1 SKIP_SUMMAC=1
cd /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
MASTER_PORT=$(( 21000 + SLURM_JOB_ID % 10000 ))
/cl/work5/hongyu-s/conda/envs/latest_transformers/bin/torchrun --nproc_per_node=2 --master_port="${MASTER_PORT}" ./run_clm.py --model_type gpt2 --tokenizer_name gpt2  --model_name_or_path /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/gpt2_medium_owt_rotary_80k/checkpoint-80000 --dataset_name mix --block_size 512 --do_train --num_train_epochs 10 --logging_steps 500 --save_steps 5000 --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --gradient_accumulation_steps 16 --learning_rate 1e-4 --weight_decay 0.0 --warmup_ratio 0.05 --attn_implementation eager --pe_method rotary --wavelet_router False --router_band_num 8 --scale_range 0 16 --wavelet_mode logit_bias_ctxscale_shift_v0 --wavelet_baseline_use False --use_beta_modulation False --use_soft_wavelet_fox False --single_A_B True --num_harmonics 1 --share_freq_across_heads True --preprocessing_num_workers 8 --ddp_timeout 21600 --seed 44 --overwrite_output_dir --output_dir /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/mix_medium_owt_rotary_10ep_s44_fp32 --logging_dir /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/mix_medium_owt_rotary_10ep_s44_fp32/tensorboard

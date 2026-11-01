#!/bin/bash
#SBATCH --job-name=PAT310_medPathFoX_b5_s42_ev
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/results_uniform/pat310_medium_pathfox/logs/%j_pathfox_b5_s42_evalwatch.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --exclude=elm72,elm26
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# Watcher: evaluates each checkpoint of pathfox_b5_s42 as soon as it is fully written (paper protocol, first 1000 examples, triton).
set -euxo pipefail
if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u; fi
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets WANDB_DISABLED=true WANDB_MODE=disabled
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID:-manual}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
RUN=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat310_medium_pathfox/pathfox_b5_s42
for STEP in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000; do
  CK=$RUN/checkpoint-$STEP
  until [ -f $CK/config.json ] && { [ -f $CK/model.safetensors ] || [ -f $CK/pytorch_model.bin ]; } && [ -f $CK/trainer_state.json ]; do sleep 120; done
  sleep 60
  for L in 512 2048 4096; do
    OUT=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/results_uniform/pat310_medium_pathfox/pathfox_b5_s42/ckpt${STEP}/L${L}; [ -f $OUT/eval_results.json ] && continue; mkdir -p $OUT
    python ./run_clm.py --model_type gpt2 --tokenizer_name gpt2 --model_name_or_path $CK --cfg_path $RUN/supply_model.cfg --dataset_name hotpot_qa --dataset_config_name distractor --hotpot_long_jsonl /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/hotpot_long_dev_uniform.jsonl --hotpot_long_lengths $L --do_eval --block_size $L --per_device_eval_batch_size 1 --max_eval_samples 1000 --path_attn_impl triton --report_to none --output_dir $OUT --overwrite_output_dir --logging_dir $OUT/log --seed 42 --attn_implementation path_attn --path_use_qk_norm false --path_use_low_rank_w true --path_use_w_shortconv false --path_conv_size 3 --path_conv_bias false --single_A_B True --use_beta_modulation False --use_soft_wavelet_fox False --wavelet_baseline_use False --use_forget_gate True --qk_rotation False --wavelet_router False --path_conv_bias false --num_harmonics 1 --ablate_switch False --load_best_model_at_end False
  done
done

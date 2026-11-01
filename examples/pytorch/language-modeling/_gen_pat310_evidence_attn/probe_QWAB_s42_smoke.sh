#!/bin/bash
#SBATCH --job-name=PAT310_evattn_QWAB_s42_smoke
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/results_uniform/pat310_evidence_attn/logs/%j_QWAB_s42_smoke.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --exclude=elm72,elm26
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
set -eo pipefail
source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets WANDB_DISABLED=true WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
for L in 512; do
  OUT=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/results_uniform/pat310_evidence_attn/QWAB_s42_smoke/L${L}; mkdir -p $OUT
  [ -f $OUT/evidence_attn.json ] && continue
  python /project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_pat310_evidence_attn/probe_evidence_attention.py --probe_out $OUT/evidence_attn.json --probe_n 5 --probe_model path -- --model_name_or_path /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/mix_medium_K1_rho256_ricker_s42/checkpoint-15000 --cfg_path /project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_pat310_evidence_attn/cfgs/QWAB_s42.cfg --model_type gpt2 --attn_implementation path_attn --path_attn_impl pytorch --path_use_qk_norm false --path_use_low_rank_w true --path_use_w_shortconv false --path_conv_size 3 --path_conv_bias false --num_harmonics 1 --single_A_B True --use_beta_modulation False --use_soft_wavelet_fox False --wavelet_baseline_use False --use_forget_gate False --qk_rotation False --ablate_switch False --wavelet_router False --bias_type wavelet --tokenizer_name gpt2 --dataset_name hotpot_qa --dataset_config_name distractor --hotpot_long_jsonl /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/hotpot_long_dev_uniform.jsonl --do_eval --per_device_eval_batch_size 1 --report_to none --overwrite_output_dir --load_best_model_at_end False --max_eval_samples 5 --hotpot_long_lengths $L --block_size $L --output_dir $OUT --logging_dir $OUT/log --seed 42
done

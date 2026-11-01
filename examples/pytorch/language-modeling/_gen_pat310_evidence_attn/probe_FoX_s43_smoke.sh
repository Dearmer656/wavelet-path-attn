#!/bin/bash
#SBATCH --job-name=PAT310_evattn_FoX_s43_smoke
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/results_uniform/pat310_evidence_attn/logs/%j_FoX_s43_smoke.txt
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
  OUT=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/results_uniform/pat310_evidence_attn/FoX_s43_smoke/L${L}; mkdir -p $OUT
  [ -f $OUT/evidence_attn.json ] && continue
  python /project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_pat310_evidence_attn/probe_evidence_attention.py --probe_out $OUT/evidence_attn.json --probe_n 5 --probe_model fox -- --model_name_or_path /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat310_fox_docorder/medium_fox_docorder_s43/checkpoint-15000 --cfg_path /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_pat310_fox_medium_docorder/cfgs/docorder.cfg --model_type forgetting_transformer --tokenizer_name gpt2 --dataset_name hotpot_qa --dataset_config_name distractor --hotpot_long_jsonl /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/hotpot_long_dev_uniform.jsonl --do_eval --per_device_eval_batch_size 1 --report_to none --overwrite_output_dir --load_best_model_at_end False --max_eval_samples 5 --hotpot_long_lengths $L --block_size $L --output_dir $OUT --logging_dir $OUT/log --seed 43
done

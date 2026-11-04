#!/bin/bash
#SBATCH --job-name=T1rep_rotary_s42_L512
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/results_pat310_table1_repro/rotary_s42/L512/%j_eval.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:p6000:2
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# PAT-310 Table 1: current-code re-eval of mix_medium_owt_rotary_10ep_s42_fp32/checkpoint-15000 at L512 with the original eval args (2 GPUs, bs 1, hotpot_long_dev.jsonl). Paper value: F1 81.16.
set -euxo pipefail
set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets WANDB_DISABLED=true WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 10000 ))
/cl/work5/hongyu-s/conda/envs/latest_transformers/bin/torchrun --nproc_per_node=2 --master_port="${MASTER_PORT}" ./run_clm.py --model_type gpt2 --tokenizer_name gpt2 --model_name_or_path /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/mix_medium_owt_rotary_10ep_s42_fp32/checkpoint-15000 --attn_implementation eager --pe_method rotary --dataset_name hotpot_qa --dataset_config_name distractor --hotpot_long_jsonl /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/hotpot_long_dev.jsonl --hotpot_long_lengths 512 --do_eval --block_size 512 --per_device_eval_batch_size 1 --output_dir /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/results_pat310_table1_repro/rotary_s42/L512 --overwrite_output_dir --logging_dir /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/results_pat310_table1_repro/rotary_s42/L512/log --load_best_model_at_end False --seed 42
python3 -c "import json; d=json.load(open('/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/results_pat310_table1_repro/rotary_s42/L512/eval_results.json')); print('REPRO rotary_s42 L512 F1=%.2f EM=%.2f n=%d (paper F1 81.16)' % (100*d['eval_f1'], 100*d['eval_em'], d['eval_samples']))"

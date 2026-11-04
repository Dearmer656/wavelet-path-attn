#!/bin/bash
#SBATCH --job-name=T1_hp_nope_s44_L2048
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/results_pat310_table1/nope_s44/L2048/%j_eval.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:p6000:4
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# PAT-310 Table 1: nope medium s44 HotpotQA-Long L2048, args of the s42 Table 1 eval (4 GPUs, bs 2, hotpot_long_dev_uniform.jsonl), full set, checkpoint-15000.
set -euxo pipefail
set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets WANDB_DISABLED=true WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SKIP_FENICE=1 SKIP_SUMMAC=1
cd /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
CKPT=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/mix_medium_owt_nope_10ep_s44/checkpoint-15000
[ -d "${CKPT}" ] || { echo "Missing ${CKPT}" >&2; exit 1; }
MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 10000 ))
/cl/work5/hongyu-s/conda/envs/latest_transformers/bin/torchrun --nproc_per_node=4 --master_port="${MASTER_PORT}" ./run_clm.py --model_type gpt2 --tokenizer_name gpt2 --model_name_or_path "${CKPT}" --pe_method no_pe --attn_implementation eager --wavelet_router False --router_band_num 8 --scale_range 0 16 --wavelet_mode logit_bias_ctxscale_shift_v0 --wavelet_baseline_use False --use_beta_modulation False --use_soft_wavelet_fox False --single_A_B True --num_harmonics 1 --share_freq_across_heads True  --dataset_name hotpot_qa --dataset_config_name distractor --hotpot_long_jsonl /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/hotpot_long_dev_uniform.jsonl --hotpot_long_lengths 2048 --do_eval --block_size 2048 --per_device_eval_batch_size 2 --output_dir /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/results_pat310_table1/nope_s44/L2048 --overwrite_output_dir --logging_dir /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/results_pat310_table1/nope_s44/L2048/log --load_best_model_at_end False --seed 42
python3 -c "import json; d=json.load(open('/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/results_pat310_table1/nope_s44/L2048/eval_results.json')); print('nope s44 L2048 F1=%.2f EM=%.2f n=%d' % (100*d['eval_f1'], 100*d['eval_em'], d['eval_samples']))"

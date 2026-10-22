#!/bin/bash
#SBATCH --job-name=PAT310_hpWPE_s43_L2048
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/results_uniform/wpe_medium_mix_10ep_s43_ckpt15000/L2048/%j_eval.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:p6000:1
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# PAT-310 (W3): WPE medium multi-seed HotpotQA-Long eval. Args copied from
# eval_hotpot_wpe_medium_mix_10ep{,_long,_16384_fullpool_1gpu}.sh (paper Table 1 WPR
# protocol: front-pinned default, bs=1, eager, checkpoint-15000), FULL sample set
# (no --max_eval_samples), L16384 on the 7404-row large_pool file.
set -euxo pipefail
set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets WANDB_DISABLED=true WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
CKPT=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/mix_medium_owt_wpe_10ep_s43/checkpoint-15000
[ -d "${CKPT}" ] || { echo "Missing ${CKPT}" >&2; exit 1; }
MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 10000 ))
/cl/work5/hongyu-s/conda/envs/latest_transformers/bin/torchrun --nproc_per_node=1 --master_port="${MASTER_PORT}" ./run_clm.py --model_type gpt2 --tokenizer_name gpt2 --model_name_or_path "${CKPT}" --pe_method wavelet --relative_type 4 --attn_implementation eager --wavelet_router False --router_band_num 8 --scale_range 0 16 --wavelet_mode logit_bias_ctxscale_shift_v0 --wavelet_baseline_use False --use_beta_modulation False --use_soft_wavelet_fox False --single_A_B True --num_harmonics 1 --share_freq_across_heads True --analyzer False --dataset_name hotpot_qa --dataset_config_name distractor --hotpot_long_jsonl /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/hotpot_long_dev_uniform.jsonl --hotpot_long_lengths 2048 --do_eval --block_size 2048 --per_device_eval_batch_size 1 --output_dir /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/results_uniform/wpe_medium_mix_10ep_s43_ckpt15000/L2048 --overwrite_output_dir --logging_dir /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/results_uniform/wpe_medium_mix_10ep_s43_ckpt15000/L2048/log --load_best_model_at_end False --seed 42
python3 -c "import json; d=json.load(open('/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/results_uniform/wpe_medium_mix_10ep_s43_ckpt15000/L2048/eval_results.json')); print('WPE s43 L2048 F1=%.4f EM=%.4f n=%d' % (d['eval_f1'], d['eval_em'], d['eval_samples']))"

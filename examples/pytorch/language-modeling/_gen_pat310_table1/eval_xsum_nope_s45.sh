#!/bin/bash
#SBATCH --job-name=T1_xsum_nope_s45
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/mix_medium_owt_nope_10ep_s45/ckpt_eval_xsum/%j_xsum_nope_medium_mix_10ep.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:p6000:2
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# PAT-310 Table 1: nope medium s45 Filtered XSum, args of the s42 Table 1 eval (2 GPUs, bs 16/8/4 at L512/1024/1536), checkpoint-15000.
set -euxo pipefail
set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets WANDB_DISABLED=true WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SKIP_FENICE=1 SKIP_SUMMAC=1
cd /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
CKPT=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/mix_medium_owt_nope_10ep_s45/checkpoint-15000
[ -d "${CKPT}" ] || { echo "Missing ${CKPT}" >&2; exit 1; }
CSV=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/mix_medium_owt_nope_10ep_s45/ckpt_eval_xsum/xsum_filter_metrics_nope_medium_mix_10ep_${SLURM_JOB_ID}.csv
echo "block_size,batch_size,theta,rouge1,rouge2,rougeL,count" > "${CSV}"
BS=(512 1024 1536); BAT=(16 8 4); TH=(0 0 0)
for i in 0 1 2; do
  OUT=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/mix_medium_owt_nope_10ep_s45/ckpt_eval_xsum/xsum_out_${SLURM_JOB_ID}/bs_${BS[$i]}; mkdir -p "${OUT}"
  if [ "${TH[$i]}" = 0 ]; then XT=""; else XT="--rope_theta ${TH[$i]}"; fi
  /cl/work5/hongyu-s/conda/envs/latest_transformers/bin/torchrun --nproc_per_node=2 --master_port=$(( 19800 + SLURM_JOB_ID % 1000 )) ./run_clm.py --model_type gpt2 --tokenizer_name gpt2 --model_name_or_path "${CKPT}" --pe_method no_pe --attn_implementation eager --wavelet_router False --router_band_num 8 --scale_range 0 16 --wavelet_mode logit_bias_ctxscale_shift_v0 --wavelet_baseline_use False --use_beta_modulation False --use_soft_wavelet_fox False --single_A_B True --num_harmonics 1 --share_freq_across_heads True  ${XT} --block_size ${BS[$i]} --dataset_name xsum --dataset_config_name default --validation_file /cl/work5/hongyu-s/fact-check-summarization/xsum_test_filter_level2_official_style.jsonl --do_eval --per_device_eval_batch_size ${BAT[$i]} --output_dir "${OUT}" --overwrite_output_dir --xsum_bucket_size 512 --xsum_bucket_apply_to eval_test --load_best_model_at_end False --seed 42
  python3 -c "import json; d=json.load(open('${OUT}/eval_results.json')); print('${BS[$i]},${BAT[$i]},${TH[$i]},%s,%s,%s,%s' % tuple(d.get(k) for k in ('eval_rouge1','eval_rouge2','eval_rougeL','eval_count')))" >> "${CSV}"
done
cat "${CSV}"

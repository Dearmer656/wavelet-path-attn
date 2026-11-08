#!/bin/bash
#SBATCH --job-name=PAT310_gateprobe15k
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/results_uniform/pat310_medium_pathfox/gate_probe/logs/%j_gateprobe15k.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:1
#SBATCH --exclude=elm72,elm26
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# Forget-gate retention probe of medium PaTH-FoX b5 s42/s43/s44 checkpoint-15000 (final), same protocol as the ckpt-5000 probe (first 200 examples, L4096/L512).
set -euxo pipefail
if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u; fi
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets WANDB_DISABLED=true WANDB_MODE=disabled
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID:-manual}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
P=/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_pat310_pathfox_gate
O=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/results_uniform/pat310_medium_pathfox/gate_probe
for s in 42 43 44; do
  RUN=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat310_medium_pathfox/pathfox_b5_s${s}; CK=$RUN/checkpoint-15000
  for L in 4096 512; do
    OUT=$O/b5_s${s}_ckpt15000_L${L}; mkdir -p $OUT
    [ -f $O/b5_s${s}_ckpt15000_L${L}.npz ] && continue
    python $P/probe_gate_retention.py --probe_out $O/b5_s${s}_ckpt15000_L${L}.npz --probe_n 200 -- --model_name_or_path $CK --cfg_path $RUN/supply_model.cfg --hotpot_long_lengths $L --block_size $L --max_eval_samples 200 --output_dir $OUT --overwrite_output_dir --logging_dir $OUT/log --model_type gpt2 --tokenizer_name gpt2 --dataset_name hotpot_qa --dataset_config_name distractor --hotpot_long_jsonl /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/hotpot_long_dev_uniform.jsonl --do_eval --per_device_eval_batch_size 1 --path_attn_impl triton --report_to none --seed ${s} --attn_implementation path_attn --path_use_qk_norm false --path_use_low_rank_w true --path_use_w_shortconv false --path_conv_size 3 --path_conv_bias false --single_A_B True --use_beta_modulation False --use_soft_wavelet_fox False --wavelet_baseline_use False --use_forget_gate True --qk_rotation False --wavelet_router False --path_conv_bias false --num_harmonics 1 --ablate_switch False --load_best_model_at_end False
  done
done
for L in 4096 512; do
  echo "== L$L (ckpt-15000)"; python $P/summarize_gate_scaled.py $O/b5_s42_ckpt15000_L$L.npz $O/b5_s43_ckpt15000_L$L.npz $O/b5_s44_ckpt15000_L$L.npz
done | tee $O/summary_scaled_ckpt15000.txt
echo "== reference: s42 ckpt-5000, same scaled summary" | tee -a $O/summary_scaled_ckpt15000.txt
python $P/summarize_gate_scaled.py $O/b5_ckpt5000_L4096.npz $O/b5_ckpt5000_L512.npz | tee -a $O/summary_scaled_ckpt15000.txt

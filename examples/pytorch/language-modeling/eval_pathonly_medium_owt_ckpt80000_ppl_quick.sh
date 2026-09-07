#!/bin/bash
#SBATCH --job-name=pathonlymed_ppl_quick
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/gpt2_medium_owt_pytorch_level_path_attn/train/%j_ppl_quick_ckpt80000.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:2
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Quick extended-length OWT perplexity spot-check for the PaTH-only (no positional bias,
# vanilla) GPT-2 medium pretrain's checkpoint-80000 -- the shared backbone that both plain
# PaTH-only baselines and QWAB's own finetune (mix_medium_owt_dd_10ep) build from. Same
# protocol as the Rotary/ALiBi checks (L1024/2048/4096/8192/12288/16384, 1000-sample cap). pe_method=vanilla
# (no explicit positional encoding -- PaTH's own data-dependent state transitions are the
# only position-sensitive mechanism), attn_implementation=path_attn with the exact
# path_* flags used at this checkpoint's own training time (per
# train_gpt2_medium_owt_mix_dd_10ep.sh's loading config).

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
CKPT="${BASE}/runs/gpt2_medium_owt_pytorch_level_path_attn/checkpoint-80000"
OUT_ROOT="${BASE}/runs/gpt2_medium_owt_pytorch_level_path_attn/ppl_quick_ckpt80000"
mkdir -p "${OUT_ROOT}"
# path_attn (unlike eager) actually reads wavelet_ctxscale_* config, so setting
# wavelet_mode=logit_bias_ctxscale_shift_v0 here (kept for CLI-flag parity with the other
# scripts even though wavelet_router=False) triggers the same scale_max_exp validation
# bug seen elsewhere in this session. Supply an explicit list to avoid it.
CFG_PATH="${OUT_ROOT}/supply_model.cfg"
echo "wavelet_ctxscale_scale_max_exp=[14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0]" > "${CFG_PATH}"
cd "${BASE}"

for BSIZE in 1024 2048 4096 8192 12288 16384; do
  OUTPUT="${BASE}/runs/gpt2_medium_owt_pytorch_level_path_attn/ppl_quick_ckpt80000/L${BSIZE}"
  mkdir -p "${OUTPUT}"
  echo "=== PaTH-only medium ckpt80000 ppl @ block_size=${BSIZE} (1000 samples) ==="
  MASTER_PORT=$(( 13300 + SLURM_JOB_ID % 10000 + BSIZE % 100 ))
  python -m torch.distributed.run --nproc_per_node=2 --master_port=${MASTER_PORT} ./run_clm.py \
    --model_type gpt2 --tokenizer_name gpt2 \
    --model_name_or_path "${CKPT}" \
    --dataset_name openwebtext \
    --validation_split_percentage 1 \
    --max_eval_samples 1000 \
    --preprocessing_num_workers 8 \
    --pe_method vanilla --attn_implementation path_attn \
    --path_use_qk_norm false \
    --path_use_low_rank_w true \
    --path_use_w_shortconv false \
    --path_conv_size 3 \
    --path_conv_bias false \
    --block_size "${BSIZE}" \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --wavelet_router False \
    --wavelet_mode logit_bias_ctxscale_shift_v0 \
    --scale_range 0 16 \
    --router_band_num 8 \
    --use_beta_modulation False \
    --use_soft_wavelet_fox False \
    --wavelet_baseline_use False \
    --single_A_B True \
    --num_harmonics 1 \
    --share_freq_across_heads True \
    --output_dir "${OUTPUT}" --overwrite_output_dir \
    --logging_dir "${OUTPUT}/log" \
    --ddp_timeout 21600 --seed 42 --load_best_model_at_end False \
    --cfg_path "${CFG_PATH}"
  python3 -c "import json; d=json.load(open('${OUTPUT}/eval_results.json')); print(f'PaTH-only medium ckpt80000 L${BSIZE} (1000 samples): eval_loss={d[\"eval_loss\"]:.4f} ppl={d[\"perplexity\"]:.2f}')"
done

echo "=== Done: PaTH-only medium ckpt80000 quick ppl (L1024-16384, 1000 samples each; longer lengths may OOM on eager/path_attn at bs=1, watch and adjust if so) ==="

#!/bin/bash
#SBATCH --job-name=qwabmed_ppl_quick
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/mix_medium_owt_dd_10ep/train/%j_ppl_quick_ckpt15000.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:2
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Quick extended-length OWT perplexity spot-check for QWAB's headline medium checkpoint
# (mix_medium_owt_dd_10ep/checkpoint-15000, s42). Same protocol as the Rotary/ALiBi/
# PaTH-only checks (L1024/2048/4096/8192/12288/16384, 1000-sample cap).
# IMPORTANT CAVEAT: unlike the other three checkpoints here (raw OWT pretrains), this
# checkpoint was FINETUNED on "mix" (HotpotQA + XSum), not OWT -- evaluating it on OWT is
# an off-distribution check (does the wavelet-augmented, mix-finetuned model still do
# reasonably on its original pretraining corpus), not an apples-to-apples "same-stage"
# comparison with the other three. Report/interpret accordingly.
# attn_implementation=path_attn with the exact wavelet/distillation flags from
# train_gpt2_medium_owt_mix_dd_10ep.sh (this checkpoint's own training config) so the
# loaded weights are used the same way they were trained.

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
CKPT="${BASE}/runs/mix_medium_owt_dd_10ep/checkpoint-15000"
OUT_ROOT="${BASE}/runs/mix_medium_owt_dd_10ep/ppl_quick_ckpt15000"
mkdir -p "${OUT_ROOT}"
# Same fix as train_qwab_medium_dd_extralen2048_s42_fp32.sh: this checkpoint predates the
# wavelet_ctxscale_k/scale_max_exp fields (absent from its own supply_model.cfg and its
# saved config.json), and the current codebase's own defaults (k=8, scale_max_exp=14.0
# scalar) fail the current codebase's own validation. Supply an explicit list.
CFG_PATH="${OUT_ROOT}/supply_model.cfg"
cat "${BASE}/runs/mix_medium_owt_dd_10ep/supply_model.cfg" > "${CFG_PATH}"
echo "wavelet_ctxscale_scale_max_exp=[14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0]" >> "${CFG_PATH}"
cd "${BASE}"

for BSIZE in 1024 2048 4096 8192 12288 16384; do
  OUTPUT="${BASE}/runs/mix_medium_owt_dd_10ep/ppl_quick_ckpt15000/L${BSIZE}"
  mkdir -p "${OUTPUT}"
  echo "=== QWAB medium ckpt15000 ppl @ block_size=${BSIZE} (1000 samples, on OWT -- off-distribution) ==="
  MASTER_PORT=$(( 13400 + SLURM_JOB_ID % 10000 + BSIZE % 100 ))
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
    --single_A_B True \
    --share_freq_across_heads True \
    --wavelet_mode logit_bias_ctxscale_shift_v0 \
    --wavelet_baseline_use False \
    --wavelet_router False \
    --use_beta_modulation False \
    --use_soft_wavelet_fox False \
    --use_forget_gate False \
    --full_fine_tune False \
    --init_theta 0.847 \
    --distill_teacher wavelet \
    --distill_in_which_layers 0 \
    --distill_freq_scale 25 \
    --scale_range 0 16 \
    --weight_alpha 0.0 \
    --loss_type cos \
    --qk_rotation False \
    --router_band_num 8 \
    --router_hidden_dim 32 \
    --rel_selection all \
    --num_harmonics 1 \
    --block_size "${BSIZE}" \
    --do_eval \
    --per_device_eval_batch_size 1 \
    --output_dir "${OUTPUT}" --overwrite_output_dir \
    --logging_dir "${OUTPUT}/log" \
    --ddp_timeout 21600 --seed 42 --load_best_model_at_end False \
    --cfg_path "${CFG_PATH}"
  python3 -c "import json; d=json.load(open('${OUTPUT}/eval_results.json')); print(f'QWAB medium ckpt15000 L${BSIZE} (1000 samples, OWT): eval_loss={d[\"eval_loss\"]:.4f} ppl={d[\"perplexity\"]:.2f}')"
done

echo "=== Done: QWAB medium ckpt15000 quick ppl on OWT (L1024-16384, 1000 samples each; longer lengths may OOM at bs=1, watch and adjust if so) ==="

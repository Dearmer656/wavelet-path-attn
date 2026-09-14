#!/bin/bash
#SBATCH --job-name=hp_dispersed_qwab_L4096
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/log_file/train/%j_hotpot_dispersed_L4096_qwab_p6000x3.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:p6000:3
#SBATCH --time=25:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Retry of QWAB's L4096 dispersed-evidence HotpotQA-Long eval. Previous attempt
# (job 588865, 2x p6000) was progressing normally (~3.8s/it, 3702 steps) but got
# CANCELLED at 3h08m (~80% through) for an external/unclear reason, not a crash.
# Rerunning on 3x p6000 per explicit request, to finish faster and with more headroom.

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
cd "${BASE}"

JSONL="${BASE}/hotpot_long/data/hotpot_long_dev_uniform.jsonl"
BSIZE=4096

CFG_QWAB_DISPERSED="${BASE}/hotpot_long/results/_tmp_cfg_qwab_dispersed_l4096_p6000x3.cfg"
cat > "${CFG_QWAB_DISPERSED}" <<'CFG'
hotpot_respect_doc_order=true
wavelet_ctxscale_k=1
wavelet_ctxscale_scale_max_exp=16.0
wavelet_mode="logit_bias_ctxscale_shift_v0"
path_attn_capture_debug_tensors=false
CFG

QWAB_CKPT="${BASE}/runs/mix_medium_owt_dd_10ep/checkpoint-15000"
QWAB_OUT="${BASE}/hotpot_long/results_uniform/mix_medium_owt_dd_10ep_s42_ckpt15000/L${BSIZE}_dispersed"
mkdir -p "${QWAB_OUT}/log"

echo "=== QWAB (mix_medium_owt_dd_10ep) s42, DISPERSED evidence, L${BSIZE} (3x p6000) ==="
MASTER_PORT=$(( 22500 + SLURM_JOB_ID % 1000 ))
python -m torch.distributed.run --nproc_per_node=3 --master_port=${MASTER_PORT} ./run_clm.py \
  --model_type gpt2 --tokenizer_name gpt2 \
  --model_name_or_path "${QWAB_CKPT}" \
  --dataset_name hotpot_qa --dataset_config_name distractor \
  --hotpot_long_jsonl "${JSONL}" \
  --hotpot_long_lengths "${BSIZE}" \
  --do_eval \
  --block_size "${BSIZE}" \
  --per_device_eval_batch_size 1 \
  --attn_implementation path_attn \
  --path_use_qk_norm false \
  --path_use_low_rank_w true \
  --path_use_w_shortconv false \
  --path_conv_size 3 \
  --path_conv_bias false \
  --single_A_B True \
  --share_freq_across_heads True \
  --pe_method vanilla \
  --num_harmonics 1 \
  --wavelet_baseline_use False \
  --wavelet_router False \
  --use_beta_modulation False \
  --use_soft_wavelet_fox False \
  --use_forget_gate False \
  --init_theta 0.847 \
  --scale_range 0 16 \
  --weight_alpha 0.0 \
  --loss_type cos \
  --qk_rotation False \
  --router_band_num 8 \
  --router_hidden_dim 32 \
  --rel_selection all \
  --output_dir "${QWAB_OUT}" --overwrite_output_dir \
  --logging_dir "${QWAB_OUT}/log" \
  --seed 42 --load_best_model_at_end False \
  --cfg_path "${CFG_QWAB_DISPERSED}"
python3 -c "import json; d=json.load(open('${QWAB_OUT}/eval_results.json')); print(f'QWAB DISPERSED L${BSIZE}: F1={d[\"eval_f1\"]:.4f} EM={d[\"eval_em\"]:.4f}')"

echo "=== Done: QWAB L4096 dispersed (3x p6000) ==="

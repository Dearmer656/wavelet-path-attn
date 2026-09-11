#!/bin/bash
#SBATCH --job-name=hp_qwab_rho256_nodetach
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_hp_qwab_rho256_nodetach_ckpt15000.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:4
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# PAT-258: HotpotQA-Long L512/2048/4096 eval for the QWAB rho256
# scratch-pretrain-backbone NODETACH mix finetune (job 584541, completed
# checkpoint-15900; using checkpoint-15000 to match this project's standard
# ckpt-15000 comparison policy against every other reported config).

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

WORKDIR=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
cd "${WORKDIR}"

RUN_DIR="${WORKDIR}/runs/mix_medium_owt_qwab_rho256_scratch_nodetach_10ep"
CKPT="${RUN_DIR}/checkpoint-15000"
CFG_PATH="${RUN_DIR}/supply_model.cfg"
DATA_DIR="${WORKDIR}/hotpot_long/data"
RESULT_BASE="${WORKDIR}/hotpot_long/results_uniform/qwab_rho256_scratch_nodetach_10ep_ckpt15000"

declare -A JSONL_FOR_LEN=(
  [512]="${DATA_DIR}/hotpot_long_dev_uniform.jsonl"
  [2048]="${DATA_DIR}/hotpot_long_dev_uniform_2048only.jsonl"
  [4096]="${DATA_DIR}/hotpot_long_dev_uniform.jsonl"
)

for L in 512 2048 4096; do
  OUT_DIR="${RESULT_BASE}/L${L}"
  mkdir -p "${OUT_DIR}"
  MASTER_PORT=$(( 29000 + SLURM_JOB_ID % 1000 + L % 1000 ))
  echo "=== QWAB rho256 nodetach ckpt15000 HotpotQA-Long L${L} ==="
  /cl/work5/hongyu-s/conda/envs/latest_transformers/bin/torchrun \
    --nproc_per_node=4 \
    --master_port="${MASTER_PORT}" \
    ./run_clm.py \
    --model_type gpt2 \
    --tokenizer_name gpt2 \
    --model_name_or_path "${CKPT}" \
    --wavelet_mode logit_bias_ctxscale_shift_v0 \
    --wavelet_baseline_use False \
    --wavelet_router False \
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
    --dataset_name hotpot_qa \
    --dataset_config_name distractor \
    --hotpot_long_jsonl "${JSONL_FOR_LEN[$L]}" \
    --hotpot_long_lengths "${L}" \
    --do_eval \
    --block_size "${L}" \
    --per_device_eval_batch_size 1 \
    --output_dir "${OUT_DIR}" \
    --overwrite_output_dir \
    --logging_dir "${OUT_DIR}/log" \
    --load_best_model_at_end False \
    --seed 42 \
    --cfg_path "${CFG_PATH}"
  echo "L=${L} done: $(cat ${OUT_DIR}/eval_results.json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'F1={d[\"eval_f1\"]:.4f} EM={d[\"eval_em\"]:.4f}')" 2>/dev/null || echo 'no results')"
done

echo "=== ALL LENGTHS DONE ==="
for L in 512 2048 4096; do
  OUT_DIR="${RESULT_BASE}/L${L}"
  echo "L=${L}: $(cat ${OUT_DIR}/eval_results.json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'F1={d[\"eval_f1\"]:.4f} EM={d[\"eval_em\"]:.4f}')" 2>/dev/null || echo 'no results')"
done

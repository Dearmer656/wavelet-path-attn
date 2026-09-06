#!/bin/bash
#SBATCH --job-name=rotyarn_hp_alllen
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_rotary_yarn_medium_s42_alllen.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:2
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Zero-shot YaRN eval on the SAME plain-Rotary medium finetune checkpoint used for the
# front-placed sweep (mix_medium_owt_rotary_10ep_s42_fp32/checkpoint-15000) -- no
# retraining, mirrors the existing RoPE-NTK convention (eval_rotary_ntk_xsum_s42.sh: reuse
# the plain rotary finetune checkpoint, just override the RoPE frequency behavior at eval
# time). yarn_factor = target_length / 512 (original_max_position_embeddings=512, the
# actual pretrain/finetune block_size) at each length, so YaRN's interpolation always
# targets "extend from the true training length to this eval length" rather than a fixed
# factor. Skips L512 (factor=1 is a no-op, already have that number: F1=0.8116).
# Compare directly against eval_rotary_medium_mix_s42_hotpot_alllen.sh's plain-RoPE numbers
# at the same lengths (F1: L2048=0.0805, L4096=0.0246, L8192=0.0198, L12288=0.0162,
# L16384=0.0270) to see whether YaRN's zero-shot frequency interpolation meaningfully
# rescues extrapolation without any finetuning.

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
CKPT="${BASE}/runs/mix_medium_owt_rotary_10ep_s42_fp32/checkpoint-15000"
cd "${BASE}"

for BSIZE in 2048 4096 8192 12288 16384; do
  if [ "${BSIZE}" -le 4096 ]; then
    JSONL="${BASE}/hotpot_long/data/hotpot_long_dev.jsonl"
  else
    JSONL="${BASE}/hotpot_long/data/hotpot_long_dev_uniform_${BSIZE}only.jsonl"
  fi
  YARN_FACTOR=$(python3 -c "print(${BSIZE}/512)")
  OUTPUT="${BASE}/hotpot_long/results/rotary_yarn_medium_s42_ckpt15000/L${BSIZE}"
  mkdir -p "${OUTPUT}/log"
  echo "=== Rotary+YaRN medium (finetuned, zero-shot) s42 HotpotQA-Long L${BSIZE} (yarn_factor=${YARN_FACTOR}) ==="
  MASTER_PORT=$(( 13000 + SLURM_JOB_ID % 10000 + BSIZE % 100 ))
  python -m torch.distributed.run --nproc_per_node=2 --master_port=${MASTER_PORT} ./run_clm.py \
    --model_type gpt2 --tokenizer_name gpt2 \
    --model_name_or_path "${CKPT}" \
    --attn_implementation eager \
    --pe_method rotary \
    --use_yarn True \
    --yarn_factor "${YARN_FACTOR}" \
    --yarn_original_max_position_embeddings 512 \
    --dataset_name hotpot_qa --dataset_config_name distractor \
    --hotpot_long_jsonl "${JSONL}" \
    --hotpot_long_lengths "${BSIZE}" \
    --do_eval \
    --block_size "${BSIZE}" \
    --per_device_eval_batch_size 1 \
    --output_dir "${OUTPUT}" --overwrite_output_dir \
    --logging_dir "${OUTPUT}/log" \
    --seed 42 --load_best_model_at_end False
  python3 -c "import json; d=json.load(open('${OUTPUT}/eval_results.json')); print(f'Rotary+YaRN medium (zero-shot) s42 L${BSIZE}: F1={d[\"eval_f1\"]:.4f} eval_loss={d[\"eval_loss\"]:.4f}')"
done

echo "=== Done: Rotary+YaRN medium (zero-shot) s42 HotpotQA-Long full-length sweep ==="

#!/bin/bash
#SBATCH --job-name=mamba2med_hp_alllen
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_mamba2_medium_s42_alllen.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:2
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Full-length HotpotQA-Long F1 sweep for the FINETUNED Mamba2 medium checkpoint
# (mix_medium_owt_mamba2_10ep_s42/checkpoint-15000 -- assumed final step by analogy with
# the Rotary/ALiBi mix-finetune counterparts, same "mix" dataset/epochs/global_bs=64; fix
# CKPT below if the actual final checkpoint differs), matching Table 4's full length set
# (L512/2048/4096/8192/12288/16384). Mamba2 is a recurrent SSM (linear-time state, no
# attention window) so it needs no YaRN/RoPE-style eval-time reconfiguration -- plain
# zero-shot sweep on the finetuned weights across all lengths, same as the ALiBi sweep.
# L512/2048/4096: front-placed hotpot_long_dev.jsonl. L8192/12288/16384: per-length
# hotpot_long_dev_uniform_{L}only.jsonl (still front-pinned in practice -- see PAT-253
# session notes on build_context_budgeted's respect_doc_order default).
# Uses mamba2_env (real CUDA causal_conv1d/mamba_ssm kernels), matching train/finetune.

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate mamba2_env; set -u
fi

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
CKPT="${BASE}/runs/mix_medium_owt_mamba2_10ep_s42/checkpoint-15000"
cd "${BASE}"

for BSIZE in 512 2048 4096 8192 12288 16384; do
  if [ "${BSIZE}" -le 4096 ]; then
    JSONL="${BASE}/hotpot_long/data/hotpot_long_dev.jsonl"
  else
    JSONL="${BASE}/hotpot_long/data/hotpot_long_dev_uniform_${BSIZE}only.jsonl"
  fi
  OUTPUT="${BASE}/hotpot_long/results/mamba2_medium_s42_ckpt15000/L${BSIZE}"
  mkdir -p "${OUTPUT}/log"
  echo "=== Mamba2 medium (finetuned) s42 HotpotQA-Long L${BSIZE} ==="
  MASTER_PORT=$(( 17000 + SLURM_JOB_ID % 10000 + BSIZE % 100 ))
  python -m torch.distributed.run --nproc_per_node=2 --master_port=${MASTER_PORT} ./run_clm.py \
    --model_type mamba2 --tokenizer_name gpt2 \
    --model_name_or_path "${CKPT}" \
    --dataset_name hotpot_qa --dataset_config_name distractor \
    --hotpot_long_jsonl "${JSONL}" \
    --hotpot_long_lengths "${BSIZE}" \
    --do_eval \
    --block_size "${BSIZE}" \
    --per_device_eval_batch_size 1 \
    --output_dir "${OUTPUT}" --overwrite_output_dir \
    --logging_dir "${OUTPUT}/log" \
    --seed 42 --load_best_model_at_end False
  python3 -c "import json; d=json.load(open('${OUTPUT}/eval_results.json')); print(f'Mamba2 medium (finetuned) s42 L${BSIZE}: F1={d[\"eval_f1\"]:.4f} eval_loss={d[\"eval_loss\"]:.4f}')"
done

echo "=== Done: Mamba2 medium (finetuned) s42 HotpotQA-Long full-length sweep ==="

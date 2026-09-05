#!/bin/bash
#SBATCH --job-name=fox_med_a6k
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat226_fox/medium_owt_a6000x4/%j_train.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:4
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# PAT-164/PAT-226: FoX (Forgetting Transformer, Lin et al.) medium pretrain on OpenWebText.
# The original PaTH-attention paper compared against FoX as an external baseline; adding it
# to this session's medium "identical data" comparison group (PA-only/Rotary/ALiBi/Mamba-2,
# all GPT-2-medium-scale on OWT: block_size=512, max_steps=80000, global_bs=64,
# warmup_ratio=0.05, lr=1e-4, weight_decay=0.01).
# Config: hidden_size=1024/num_hidden_layers=24/num_heads=16 mirrors GPT-2-medium's shape
# (FoX's own default is 2048/24/32). Verified via a 30-step wikitext-103 smoke test
# (job 577316): param count 360.2M (medium class), finite loss, checkpoint round-trip OK,
# runs fine in latest_transformers (forgetting_attn is pure Triton, unlike Mamba-2's
# mamba_ssm/causal_conv1d which needed a dedicated mamba2_env). The smoke test only used
# per_device_train_batch_size=4 (single GPU); bs=8 here is a conservative starting point
# (not empirically validated at this batch/GPU-count) rather than jumping straight to
# Mamba-2's bs=16 -- eager-style attention (Rotary) needed bs=4 vs path_attn's bs=8/16 for
# the same OOM-class reason (O(T^2) backward memory), and forgetting_attn's actual memory
# profile hasn't been checked here.
# NOTE: config_overrides omits pad_token_id -- ForgettingTransformerConfig defaults it to
# None, and HF's update_from_string refuses to type-update a None-valued field via CLI
# override (see sbatch_fox_medium_smoke.sh's fix comment for the exact error). Not needed
# for this packed-sequence pretrain anyway.

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi

WORKDIR=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
cd "${WORKDIR}"

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_OUT="${WORKDIR}/runs/pat226_fox/medium_owt_a6000x4"
mkdir -p "${RUN_OUT}/train"
MASTER_PORT=$(( 24226 + SLURM_JOB_ID % 1000 ))

echo "================= BEGIN RUN PAT-226 FoX medium(360M) OWT pretrain: 4xa6000 ================="

python -m torch.distributed.run --nproc_per_node=4 --master_port=${MASTER_PORT} ./run_clm.py \
  --model_type forgetting_transformer --tokenizer_name gpt2 \
  --config_overrides "hidden_size=1024,num_hidden_layers=24,num_heads=16,vocab_size=50257,tie_word_embeddings=True,bos_token_id=50256,eos_token_id=50256" \
  --dataset_name openwebtext --validation_split_percentage 1 \
  --block_size 512 \
  --do_train --do_eval \
  --max_steps 80000 \
  --eval_strategy steps --eval_steps 5000 \
  --load_best_model_at_end True --metric_for_best_model eval_loss --greater_is_better False \
  --logging_dir "${RUN_OUT}/train_log" --logging_steps 500 \
  --save_steps 10000 --save_safetensors False \
  --per_device_train_batch_size 8 --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-4 --weight_decay 0.01 \
  --warmup_ratio 0.05 --bf16 True \
  --preprocessing_num_workers 8 \
  --output_dir "${RUN_OUT}" --overwrite_output_dir \
  --seed 42

echo "=== PAT-226 FoX medium OWT pretrain done (4xa6000) ==="

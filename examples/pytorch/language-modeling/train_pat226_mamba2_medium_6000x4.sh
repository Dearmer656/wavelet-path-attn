#!/bin/bash
#SBATCH --job-name=mamba2_med_6k
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat226_mamba2/medium_owt_6000x4/%j_train.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:4
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# PAT-164/PAT-226: Mamba-2 medium (370M ladder step) pretrain on OpenWebText, matching the
# "identical data" protocol of the GPT-2-medium PA-only/Rotary/ALiBi OWT pretrains this
# session (block_size=512, max_steps=80000, global_bs=64, warmup_ratio=0.05, do_eval +
# load_best_model_at_end, seed=42) -- NOT the 130M small-model script's wt103/epoch-based
# recipe. Deltas from the 130M PAT-226 script (train_pat226_mamba2_wt103_s42.sh):
#   - config scaled to 370M: hidden_size 768->1024, num_hidden_layers 24->48, num_heads 24->32
#     (state_size=128, expand=2, head_dim=64 unchanged -- official Mamba-2 ladder only scales
#     hidden_size/num_hidden_layers between rungs)
#   - dataset_name openwebtext (not wikitext-103), max_steps=80000 (not num_train_epochs=30)
#   - added do_eval/eval_strategy/eval_steps=5000/load_best_model_at_end/metric_for_best_model
#     (130M pretrain had eval but no best-model selection; medium protocol has it)
#   - weight_decay=0.01 (was 0.0 in 130M) -- explicitly confirmed by user to match the
#     medium GPT-2 pretrain baselines' convention rather than carry over Mamba's own 0.0
#   - per_device_train_batch_size kept at 16 (same as 130M) per user instruction ("二倍bs",
#     i.e. don't preemptively halve to 8) -- 16x4x1=64 global_bs. Untested at 370M scale;
#     watch the first few hundred steps for OOM/grad-norm blowup.
# Parallel run on 6000/p6000 (train_pat226_mamba2_medium_6000x4.sh /
# ..._p6000x4.sh) to compare stability/speed across hardware; p6000 (Quadro P6000, Pascal)
# is untested with mamba_ssm/causal_conv1d at this project -- may simply fail to build/run.
# Uses mamba2_env (real CUDA causal_conv1d/mamba_ssm kernels) -- NOT latest_transformers --
# the fla Triton conv1d fallback caused a gradient-explosion divergence at 130M scale.

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate mamba2_env; set -u
fi

WORKDIR=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
cd "${WORKDIR}"

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_OUT="${WORKDIR}/runs/pat226_mamba2/medium_owt_6000x4"
mkdir -p "${RUN_OUT}/train"
MASTER_PORT=$(( 24226 + SLURM_JOB_ID % 1000 ))

echo "================= BEGIN RUN PAT-226 mamba2 medium(370M) OWT pretrain: 4x6000 ================="

python -m torch.distributed.run --nproc_per_node=4 --master_port=${MASTER_PORT} ./run_clm.py \
  --model_type mamba2 --tokenizer_name gpt2 \
  --config_overrides "hidden_size=1024,num_hidden_layers=48,state_size=128,expand=2,head_dim=64,num_heads=32,vocab_size=50257,tie_word_embeddings=True,bos_token_id=50256,eos_token_id=50256,pad_token_id=50256" \
  --dataset_name openwebtext --validation_split_percentage 1 \
  --block_size 512 \
  --do_train --do_eval \
  --max_steps 80000 \
  --eval_strategy steps --eval_steps 5000 \
  --load_best_model_at_end True --metric_for_best_model eval_loss --greater_is_better False \
  --logging_dir "${RUN_OUT}/train_log" --logging_steps 500 \
  --save_steps 10000 --save_safetensors False \
  --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 --weight_decay 0.01 \
  --warmup_ratio 0.05 --bf16 True --tf32 True \
  --preprocessing_num_workers 8 \
  --output_dir "${RUN_OUT}" --overwrite_output_dir \
  --seed 42

echo "=== PAT-226 mamba2 medium OWT pretrain done (4x6000) ==="

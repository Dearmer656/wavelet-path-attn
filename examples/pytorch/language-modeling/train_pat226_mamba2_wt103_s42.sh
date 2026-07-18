#! /bin/bash
#SBATCH --job-name=PAT226_mamba2_pre
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat226_mamba2/wt103_s42/train/%j_pat226_mamba2_wt103.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:4
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# PAT-226: Mamba-2 130M pretrain on wikitext-103 — mirrors the PaTH backbone
# recipe (runs/1r_baseline_from_s: wt103, block 512, global bs 64, lr 1e-4,
# 30 epochs; the mix finetune later starts from checkpoint-80000).
# Config = published Mamba-2 130M ladder + GPT-2 tokenizer; num_heads=24 must
# be explicit (derived in __init__ before config_overrides apply).
# Branch: hongyusaatitech/pat-226-ssm-external-baseline-mamba-2-130m-under-the-identical-data

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate mamba2_env; set -u
fi

WORKDIR=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
cd "${WORKDIR}"

# PAT-226: dedicated env (mamba2_env, torch 2.7.0+cu126) with the REAL CUDA
# causal_conv1d/mamba_ssm kernels (matching cu12torch2.7 prebuilt wheels).
# Fixes the gradient-explosion divergence traced to fla's untested Triton
# conv1d fallback (see PAT-226 Linear comments, 2026-07-18); verified stable
# via a 20-step smoke test (grad_norm 10.88->10.54, no blowup).
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_OUT="${WORKDIR}/runs/pat226_mamba2/wt103_s42"
mkdir -p "${RUN_OUT}/train"
MASTER_PORT=$(( 24226 + SLURM_JOB_ID % 1000 ))

echo "================= BEGIN RUN PAT-226 mamba2 wt103 pretrain s42 ================="

python -m torch.distributed.run --nproc_per_node=4 --master_port=${MASTER_PORT} ./run_clm.py \
  --model_type mamba2 --tokenizer_name gpt2 \
  --config_overrides "hidden_size=768,num_hidden_layers=24,state_size=128,expand=2,head_dim=64,num_heads=24,vocab_size=50257,tie_word_embeddings=True,bos_token_id=50256,eos_token_id=50256,pad_token_id=50256" \
  --learning_rate 1e-4 --weight_decay 0.0 \
  --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
  --block_size 512 --dataset_name wikitext --dataset_config_name wikitext-103-raw-v1 \
  --do_train --do_eval --eval_strategy steps --eval_steps 2500 \
  --logging_dir "${RUN_OUT}/train_log" --logging_steps 500 \
  --num_train_epochs 30 \
  --save_steps 2500 --save_safetensors False \
  --warmup_ratio 0.05 --bf16 True --tf32 True \
  --output_dir "${RUN_OUT}" --overwrite_output_dir \
  --seed 42

echo "=== PAT-226 mamba2 wt103 pretrain done ==="

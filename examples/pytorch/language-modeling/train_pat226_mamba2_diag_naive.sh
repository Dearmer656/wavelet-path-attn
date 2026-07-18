#! /bin/bash
#SBATCH --job-name=PAT226_diagA
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat226_mamba2/diag_naive/train/%j.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:4
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# PAT-226 diagnostic A: 2000-step wt103 pretrain smoke with FLA_MAMBA2_FORCE_NAIVE=1
# (pure PyTorch forward, bypasses the untested Triton conv1d fast path). If
# grad_norm stays bounded here (vs exploding by step 1500-2000 as in the
# original run), the Triton conv1d kernel is confirmed as the divergence cause.

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
export FLA_MAMBA2_FORCE_NAIVE=1

RUN_OUT="${WORKDIR}/runs/pat226_mamba2/diag_naive"
mkdir -p "${RUN_OUT}/train"
MASTER_PORT=$(( 24401 + SLURM_JOB_ID % 1000 ))

python -m torch.distributed.run --nproc_per_node=4 --master_port=${MASTER_PORT} ./run_clm.py \
  --model_type mamba2 --tokenizer_name gpt2 \
  --config_overrides "hidden_size=768,num_hidden_layers=24,state_size=128,expand=2,head_dim=64,num_heads=24,vocab_size=50257,tie_word_embeddings=True,bos_token_id=50256,eos_token_id=50256,pad_token_id=50256" \
  --learning_rate 1e-4 --weight_decay 0.0 \
  --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
  --block_size 512 --dataset_name wikitext --dataset_config_name wikitext-103-raw-v1 \
  --do_train --max_steps 2000 \
  --logging_steps 100 \
  --save_steps 100000 --save_safetensors False \
  --warmup_ratio 0.05 --bf16 True --tf32 True \
  --output_dir "${RUN_OUT}" --overwrite_output_dir \
  --seed 42

echo "=== PAT-226 diag A (force naive) done ==="

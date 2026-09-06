#!/bin/bash
#SBATCH --job-name=fox_med_smoke_ddp4
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/log_file/train/%j_fox_medium_smoke_ddp4.txt
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:a6000:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00

# Debug repro: identical to sbatch_fox_medium_smoke.sh (30 steps, wikitext-103, same
# config), but launched via torch.distributed.run --nproc_per_node=4 instead of plain
# `python`, matching the real pretrain script's launch mechanism. The single-process
# smoke test (job 577432) showed correct loss (9.49->8.43->8.02); the real 4-GPU DDP
# pretrain (job 577434) showed the ORIGINAL broken pattern (64.3->49.25->44.09...) despite
# the "[FoX] re-applied _init_weights" confirmation log line firing. This isolates whether
# the DDP launch mechanism itself is what breaks the init fix.

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

RUN_OUT="${WORKDIR}/runs/pat226_fox/smoke_ddp4"
mkdir -p "${RUN_OUT}"
MASTER_PORT=$(( 24300 + SLURM_JOB_ID % 1000 ))

python -m torch.distributed.run --nproc_per_node=4 --master_port="${MASTER_PORT}" ./run_clm.py \
  --model_type forgetting_transformer --tokenizer_name gpt2 \
  --config_overrides "hidden_size=1024,num_hidden_layers=24,num_heads=16,vocab_size=50257,tie_word_embeddings=True,bos_token_id=50256,eos_token_id=50256" \
  --learning_rate 1e-4 --weight_decay 0.01 \
  --per_device_train_batch_size 4 --per_device_eval_batch_size 4 \
  --block_size 512 --dataset_name wikitext --dataset_config_name wikitext-103-raw-v1 \
  --do_train --max_steps 30 --save_steps 30 --save_safetensors False \
  --logging_steps 5 \
  --output_dir "${RUN_OUT}" --overwrite_output_dir \
  --seed 42

echo "=== PAT-226 FoX medium smoke (DDP4) done ==="

#! /bin/bash
#SBATCH --job-name=PAT226_mamba2_mix
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat226_mamba2/mix_s42/train/%j_pat226_mamba2_mix.txt
#SBATCH --partition=lang_gpu_long
#SBATCH --account=lang
#SBATCH --gres=gpu:a100:4
#SBATCH --nodelist=lang01
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# PAT-226 stage 2: Mamba-2 130M mix finetune from wt103 ckpt-80000.
# Mirrors the canonical A4 mix finetune protocol (10 epochs, global bs 64,
# lr 1e-4, block 512, seed 42) minus PaTH/wavelet-specific flags.
# Precision matches the PAT-226 pretrain (bf16 + tf32) — disclosed in paper.
# Branch: hongyusaatitech/pat-226-ssm-external-baseline-mamba-2-130m-under-the-identical-data

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

RUN_OUT="${WORKDIR}/runs/pat226_mamba2/mix_s42"
mkdir -p "${RUN_OUT}/train"
MASTER_PORT=$(( 24326 + SLURM_JOB_ID % 1000 ))

echo "================= BEGIN RUN PAT-226 mamba2 mix finetune s42 ================="

python -m torch.distributed.run --nproc_per_node=4 --master_port=${MASTER_PORT} ./run_clm.py \
  --model_name_or_path "${WORKDIR}/runs/pat226_mamba2/wt103_s42/checkpoint-80000" \
  --tokenizer_name gpt2 \
  --learning_rate 1e-4 --weight_decay 0.0 \
  --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
  --block_size 512 --dataset_name mix \
  --do_train --do_eval --eval_strategy steps --eval_steps 500 \
  --logging_dir "${RUN_OUT}/train_log" --logging_steps 500 \
  --num_train_epochs 10 \
  --save_steps 2500 --save_safetensors False \
  --warmup_ratio 0.05 --bf16 True --tf32 True \
  --output_dir "${RUN_OUT}" --overwrite_output_dir \
  --gradient_accumulation_steps 1 \
  --seed 42

echo "=== PAT-226 mamba2 mix finetune done ==="

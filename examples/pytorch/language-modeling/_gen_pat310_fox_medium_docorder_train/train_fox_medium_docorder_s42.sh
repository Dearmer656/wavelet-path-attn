#!/bin/bash
#SBATCH --job-name=PAT310_foxmed_doctrain_s42
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat310_fox_docorder/medium_fox_docorder_s42/train/%j_train.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:p6000:2
#SBATCH --exclude=elm72
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# PAT-310 (W5): medium FoX mix finetune identical to train_pat226_fox_medium_mix_10ep_s42_a6000x4.sh, except the HotpotQA
# evidence is kept at its original document position (hotpot_respect_doc_order=True), matching the medium PaTH s43/s44 and
# QWAB finetunes (08-19..09-01). Chains the doc-order HotpotQA-Long L8192/12288/16384 evals.
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

PRETRAIN_CKPT="${WORKDIR}/runs/pat226_fox/medium_owt_p6000x2_fixed/checkpoint-80000"
OUT="${WORKDIR}/runs/pat310_fox_docorder/medium_fox_docorder_s42"
mkdir -p "${OUT}"

MASTER_PORT=$(( 24900 + SLURM_JOB_ID % 1000 ))

echo "=== PAT-310 medium FoX doc-order mix finetune s42 ==="
[ -d "${PRETRAIN_CKPT}" ] || { echo "Missing ${PRETRAIN_CKPT} -- did the pretrain job actually complete?" >&2; exit 1; }

python -m torch.distributed.run --nproc_per_node=2 --master_port=${MASTER_PORT} ./run_clm.py \
  --model_type forgetting_transformer --tokenizer_name gpt2 \
  --model_name_or_path "${PRETRAIN_CKPT}" \
  --dataset_name mix \
  --block_size 512 \
  --do_train \
  --num_train_epochs 10 \
  --logging_steps 500 \
  --save_steps 5000 \
  --save_safetensors False \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-4 \
  --weight_decay 0.0 \
  --warmup_ratio 0.05 \
  --preprocessing_num_workers 8 \
  --ddp_timeout 21600 \
  --cfg_path /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_pat310_fox_medium_docorder/cfgs/docorder.cfg \
  --seed 42 \
  --overwrite_output_dir \
  --output_dir "${OUT}" \
  --logging_dir "${OUT}/tensorboard"

[ -d "${OUT}/checkpoint-15000" ] || { echo "Missing final checkpoint" >&2; exit 1; }
for L in 8192 12288 16384; do sbatch /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_pat310_fox_medium_docorder_train/eval_fox_medium_docordertrain_s42_L${L}.sh; done

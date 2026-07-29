#! /bin/bash
#SBATCH --job-name=PAT226_mamba2_bestckpt
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat226_mamba2/wt103_s42_bestckpt/train/%j_pat226_mamba2_wt103_bestckpt.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:4
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# PAT-226 baseline fix: same recipe as train_pat226_mamba2_wt103_s42.sh
# (wt103, block 512, global bs 64, lr 1e-4, 30 epochs) but with
# load_best_model_at_end + early stopping so the reported checkpoint is the
# eval-loss optimum (~epoch 7, ppl~24 in the original run) instead of the
# epoch-30 overfit point (ppl~70). Root cause: Mamba2Config has no dropout
# field at all (unlike the GPT2/PaTH baseline's attn/embd/resid_pdrop=0.1),
# so it has no architectural regularization to fall back on; weight_decay is
# NOT added here because HF Trainer's get_decay_parameter_names only excludes
# params whose name matches bias/norm patterns — it would decay A_log/D
# (the SSM's own `_no_weight_decay` marking on these is not read by HF
# Trainer at all), which the official Mamba2 recipe never does. Fixing that
# safely needs a custom optimizer param-group override, out of scope for
# this baseline-fix pass.
# Branch: hongyusaatitech/pat-226-ssm-external-baseline-mamba-2-130m-under-the-identical-data

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

RUN_OUT="${WORKDIR}/runs/pat226_mamba2/wt103_s42_bestckpt"
mkdir -p "${RUN_OUT}/train"
MASTER_PORT=$(( 24226 + SLURM_JOB_ID % 1000 ))

echo "================= BEGIN RUN PAT-226 mamba2 wt103 pretrain s42 (bestckpt fix) ================="

python -m torch.distributed.run --nproc_per_node=4 --master_port=${MASTER_PORT} ./run_clm.py \
  --model_type mamba2 --tokenizer_name gpt2 \
  --config_overrides "hidden_size=768,num_hidden_layers=24,state_size=128,expand=2,head_dim=64,num_heads=24,vocab_size=50257,tie_word_embeddings=True,bos_token_id=50256,eos_token_id=50256,pad_token_id=50256" \
  --learning_rate 1e-4 --weight_decay 0.0 \
  --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
  --block_size 512 --dataset_name wikitext --dataset_config_name wikitext-103-raw-v1 \
  --do_train --do_eval --eval_strategy steps --eval_steps 2500 \
  --logging_dir "${RUN_OUT}/train_log" --logging_steps 500 \
  --num_train_epochs 30 \
  --save_strategy steps --save_steps 2500 --save_safetensors False --save_total_limit 3 \
  --load_best_model_at_end True --metric_for_best_model eval_loss --greater_is_better False \
  --early_stopping_patience 8 \
  --warmup_ratio 0.05 --bf16 True --tf32 True \
  --output_dir "${RUN_OUT}" --overwrite_output_dir \
  --seed 42

echo "=== PAT-226 mamba2 wt103 pretrain (bestckpt fix) done ==="

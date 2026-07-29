#! /bin/bash
#SBATCH --job-name=PAT226_mamba2_bestckpt
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat226_mamba2/wt103_s42_bestckpt/train/%j_pat226_mamba2_wt103_bestckpt.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:2
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# PAT-226 baseline fix v2: actively fight the overfitting instead of just
# rescuing the best checkpoint after the fact. v1 (load_best_model_at_end +
# patience=8, weight_decay=0.0) let the run continue climbing from ppl 24.1
# (step 25000) back up to ppl 26.3 by step 32500 before intervening -
# wasteful and not a real fix, just a rescue. Root cause is still: Mamba2Config
# has no dropout field at all (unlike GPT2/PaTH's attn/embd/resid_pdrop=0.1).
# Changes in v2:
#  - weight_decay 0.1 (Mamba paper's own AdamW convention), now safe: run_clm.py
#    got a SSMAwareTrainer that extends get_decay_parameter_names to also
#    exclude A_log/D (HF's default only excludes bias/norm-named params, so a
#    naive weight_decay would have decayed the SSM's own dynamics parameters,
#    which the official recipe never does).
#  - early_stopping_patience 8 -> 4 (10000 steps of tolerance instead of
#    20000) so training actually halts once overfitting is clearly underway,
#    instead of grinding through it for another 5+ epochs.
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

python -m torch.distributed.run --nproc_per_node=2 --master_port=${MASTER_PORT} ./run_clm.py \
  --model_type mamba2 --tokenizer_name gpt2 \
  --config_overrides "hidden_size=768,num_hidden_layers=24,state_size=128,expand=2,head_dim=64,num_heads=24,vocab_size=50257,tie_word_embeddings=True,bos_token_id=50256,eos_token_id=50256,pad_token_id=50256" \
  --learning_rate 1e-4 --weight_decay 0.1 \
  --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --gradient_accumulation_steps 2 \
  --block_size 512 --dataset_name wikitext --dataset_config_name wikitext-103-raw-v1 \
  --do_train --do_eval --eval_strategy steps --eval_steps 2500 \
  --logging_dir "${RUN_OUT}/train_log" --logging_steps 500 \
  --num_train_epochs 30 \
  --save_strategy steps --save_steps 2500 --save_safetensors False --save_total_limit 3 \
  --load_best_model_at_end True --metric_for_best_model eval_loss --greater_is_better False \
  --early_stopping_patience 4 \
  --warmup_ratio 0.05 --bf16 True --tf32 True \
  --output_dir "${RUN_OUT}" --overwrite_output_dir \
  --seed 42

echo "=== PAT-226 mamba2 wt103 pretrain (bestckpt fix) done ==="

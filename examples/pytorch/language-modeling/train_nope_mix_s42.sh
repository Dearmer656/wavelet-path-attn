#!/bin/bash
#SBATCH --job-name=nope_mix_s42
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/nope_mix_finetune/s42/%j_train_nope_mix_s42.txt
#SBATCH --partition=gpu_long
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:3090:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# Clean NoPE external baseline for Table 4 (GPT-2 small), matching ALiBi/RoPE's exact recipe.
# Base pretrain: runs/wikitext_pe_cmp/nope/checkpoint-80000 — genuine no_pe FROM STEP 0
# (verified in code: modeling_gpt2.py never instantiates self.wpe and never adds position_embeds
# when pe_method in ('no_pe','wavelet','alibi'); this pretrain script is a documented
# "one-factor delta from train_rotary_wt103.sh: pe_method=no_pe (was rotary)").
# This REPLACES the previously-used runs/wikitext_pe_cmp/wavelet/finetune_eager_nope_seed42
# checkpoint, whose PRETRAIN stage actually used wavelet PE (only stripped to no_pe at finetune
# time) — not a fair "no positional encoding at all" baseline. See PAT-256 for the full writeup.
# GPU: rerouted from a6000x4 to elm26's idle 6000x2 (no a6000 free); per_device_bs=16 unchanged,
# gradient_accumulation_steps doubled 1->2 to keep global_bs=64 (16*2gpu*2accum=64).

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true

WORKDIR=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
cd "${WORKDIR}"

MASTER_PORT=$(( 19000 + SLURM_JOB_ID % 1000 ))
PRETRAIN_CKPT="${WORKDIR}/runs/wikitext_pe_cmp/nope/checkpoint-80000"
OUT_DIR="${WORKDIR}/runs/nope_mix_finetune/s42"
mkdir -p "${OUT_DIR}"

cat > "${WORKDIR}/runs/nope_mix_finetune/supply_model.cfg" <<'CFG'
router_mode="seperate"
coe_mode="none"
tau=1
scale_type="none"
hotpot_question_position="later"
CFG

python -m torch.distributed.run --nproc_per_node=2 --master_port="${MASTER_PORT}" ./run_clm.py \
  --model_name_or_path "${PRETRAIN_CKPT}" \
  --tokenizer_name gpt2 \
  --dataset_name mix \
  --pe_method no_pe \
  --attn_implementation eager \
  --block_size 512 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --num_train_epochs 10 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --do_train \
  --do_eval \
  --eval_strategy steps \
  --eval_steps 5000 \
  --logging_steps 250 \
  --save_steps 5000 \
  --save_total_limit 10 \
  --output_dir "${OUT_DIR}" \
  --overwrite_output_dir \
  --share_freq_across_heads True \
  --wavelet_router False \
  --wavelet_mode logit_bias_ctxscale_shift_v0 \
  --scale_range 0 16 \
  --analyzer False \
  --router_band_num 8 \
  --use_beta_modulation False \
  --use_soft_wavelet_fox False \
  --wavelet_baseline_use False \
  --single_A_B True \
  --num_harmonics 1 \
  --cfg_path "${WORKDIR}/runs/nope_mix_finetune/supply_model.cfg" \
  --seed 42

echo "Done: NoPE (clean, from-scratch) mix finetune s42"

#!/bin/bash
# PAT-310 W4: L_train=256 rho sweep (64/128/256 = 0.25/0.5/1.0 L_train, mirrors the paper's 128/256/512 at L_train=512) + PaTH baseline, 3 seeds.
# Eval protocol = paper Appendix A (_gen_headline_extrap_curve): hotpot_long_dev_uniform.jsonl, front-placed default, 2 GPU x eval bs 2, pytorch impl, ckpt-15000.
set -e
D=/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_pat310_L256train
LM=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
RES=$LM/hotpot_long/results_uniform/pat310_L256train
HDR='set -euxo pipefail
if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u; fi
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true WANDB_MODE=disabled'
mkeval () { # name ckpt_dir gres
  local N=$1 R=$2 G=$3
  cat > $D/eval_${N}.sh <<EOS
#!/bin/bash
#SBATCH --job-name=L256tr_eval_${N}
#SBATCH --output=$RES/logs/%j_eval_${N}.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:${G}:2
#SBATCH --exclude=elm72,elm26
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
$HDR
JSONL="$LM/hotpot_long/data/hotpot_long_dev_uniform.jsonl"
CHECKPOINT="$R/checkpoint-15000"
CFG_PATH="$R/supply_model.cfg"
[ -d "\${CHECKPOINT}" ] || { echo "Missing \${CHECKPOINT}" >&2; exit 1; }
cd "$LM"
for BLOCK_SIZE in 512 2048 4096; do  # hotpot_long_dev_uniform.jsonl only holds the paper lengths 512/2048/4096
OUTPUT_DIR="$RES/${N}_ckpt15000/L\${BLOCK_SIZE}"
[ -f "\${OUTPUT_DIR}/eval_results.json" ] && continue
mkdir -p "\${OUTPUT_DIR}"
MASTER_PORT=\$((12000 + SLURM_JOB_ID % 10000))
torchrun --nproc_per_node=2 --master_port=\${MASTER_PORT} ./run_clm.py --model_type gpt2 --tokenizer_name gpt2 --model_name_or_path "\${CHECKPOINT}" --attn_implementation path_attn --cfg_path "\${CFG_PATH}" --dataset_name hotpot_qa --dataset_config_name distractor --hotpot_long_jsonl "\${JSONL}" --hotpot_long_lengths \${BLOCK_SIZE} --do_eval --block_size \${BLOCK_SIZE} --per_device_eval_batch_size 2 --path_attn_impl pytorch --report_to none --output_dir "\${OUTPUT_DIR}" --overwrite_output_dir --logging_dir "\${OUTPUT_DIR}/log" --seed 42 --path_use_qk_norm false --path_use_low_rank_w true --path_use_w_shortconv false --path_conv_size 3 --path_conv_bias false --num_harmonics 1 --single_A_B True --use_beta_modulation False --use_soft_wavelet_fox False --wavelet_baseline_use False --use_forget_gate False --qk_rotation False --ablate_switch False --wavelet_router False --load_best_model_at_end False
done
echo "=== Done: ${N} ==="
EOS
}
Q=$LM/runs/pat244_dual_temp
for s in 42 43 44; do
  mkeval rho64_s$s  $Q/K1_L256_me12_rho64_ricker_s$s a6000
  mkeval rho128_s$s $Q/K1_L256_me14_rho128_ricker_s$s a6000
done
mkeval rho256_s42 $Q/K1_L256_me16_rho256_ricker_s42 a6000
mkeval rho256_s43 $Q/K1_L256_me16_rho256_seed43 a6000
mkeval rho256_s44 $Q/K1_L256_me16_rho256_ricker_s44 a6000
for s in 42 43 44; do
  RO=$LM/runs/pat310_L256train/PaTH_s$s
  mkeval PaTH_s$s $RO a6000
  G=p6000; [ $s = 42 ] && G=a100-80
  cat > $D/train_PaTH_s$s.sh <<EOS
#!/bin/bash
#SBATCH --job-name=L256tr_PaTH_s$s
#SBATCH --output=$RO/train/%j_train.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:${G}:2
#SBATCH --exclude=elm72,elm26
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# Copy of _gen_postfix_4way_s42_L256/train_PA_only_s42_4gpu_L256.sh (same init runs/1r_baseline_from_s/checkpoint-80000 as the QWAB L256 runs,
# block 256, lr 1e-4, warmup 0.05, 10 ep, global bs 64) with: seed $s, 2 GPU x bs16 x accum 2, --path_attn_impl pytorch (as the QWAB L256 runs), chained eval.
$HDR
cd "$LM"
RUN_OUT="$RO"
MASTER_PORT=\$((12000 + SLURM_JOB_ID % 20000))
mkdir -p "\${RUN_OUT}/train"
cat > "\${RUN_OUT}/supply_model.cfg" <<'CFG'
router_mode="seperate"
coe_mode="none"
hotpot_question_position="later"
wavelet_mode="off"
wavelet_ctxscale_k=1
wavelet_ctxscale_scale_max_exp=14
CFG
python -m torch.distributed.run --nproc_per_node=2 --master_port="\${MASTER_PORT}" ./run_clm.py --model_type gpt2 --tokenizer_name gpt2 --config_name gpt2 --share_freq_across_heads True --learning_rate 1e-4 --weight_decay 0.0 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --block_size 256 --dataset_name mix --do_train --eval_strategy no --logging_dir "\${RUN_OUT}/train_log" --logging_steps 500 --num_train_epochs 10 --num_harmonics 1 --wavelet_pe_softmax_use False --save_steps 2500 --attn_implementation path_attn --path_attn_impl pytorch --path_use_qk_norm false --path_use_low_rank_w true --path_use_w_shortconv false --path_conv_size 3 --warmup_ratio 0.05 --path_conv_bias false --output_dir "\${RUN_OUT}" --overwrite_output_dir --gradient_accumulation_steps 2 --pe_method no_pe --single_A_B True --use_beta_modulation False --use_soft_wavelet_fox False --wavelet_mode off --model_name_or_path runs/1r_baseline_from_s/checkpoint-80000 --full_fine_tune False --wavelet_baseline_use False --init_theta 0.847 --use_forget_gate False --qk_rotation False --wavelet_router False --distill_in_which_layers 0 --spectral_loss_coe 0.0 --temp_loss_coe 0.0 --cfg_path "\${RUN_OUT}/supply_model.cfg" --seed $s
[ -d "\${RUN_OUT}/checkpoint-15000" ] || { echo "Missing final checkpoint" >&2; exit 1; }
sbatch $D/eval_PaTH_s$s.sh
EOS
done

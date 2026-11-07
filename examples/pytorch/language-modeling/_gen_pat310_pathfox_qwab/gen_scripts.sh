#!/bin/bash
# PaTH-FoX + QWAB on GPT-2 medium: QWAB rho256 recipe (_gen_pat244_medium/train_mix_medium_K1_rho256_ricker_s42.sh)
# plus the forget gate at bias 5 (as PaTH-FoX b5), both added at fine-tuning to the same PaTH checkpoint.
# 2 GPUs at the same per-device batch with doubled accumulation (global batch 64, as the 4-GPU QWAB runs).
# Eval: PyTorch at L<=4096 (gate + bias); Triton at L4096..16384, the paper's QWAB protocol there, where the bias is absent.
set -euo pipefail
W=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
G=$W/_gen_pat310_pathfox_qwab
# B = forget-gate bias init (5 = the original runs, unchanged names); other values get a _b$B suffix.
B=${B:-5}; SEEDS=${SEEDS:-"42 43 44"}; TRAIN_GRES=${TRAIN_GRES:-p6000:2}
SUF=$([ "$B" = 5 ] && echo "" || echo "_b$B")
HEAD='set -euxo pipefail
set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets WANDB_DISABLED=true WANDB_MODE=disabled
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID:-manual}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
MASTER_PORT=$(( 24000 + SLURM_JOB_ID % 5000 ))'

sbatch_head() {  # name gres output
  printf '#!/bin/bash\n#SBATCH --job-name=%s\n#SBATCH --output=%s\n#SBATCH --partition=gpu_long\n#SBATCH --gres=gpu:%s\n#SBATCH --exclude=elm72,elm26\n#SBATCH --time=100:00:00\n#SBATCH --ntasks=1\n#SBATCH --cpus-per-task=4\n' "$1" "$3" "$2"
}

gen_eval() {  # seed tag impl L nproc bs gres jsonl
  local s=$1 tag=$2 impl=$3 L=$4 np=$5 bs=$6 gres=$7 jsonl=$8
  local RUN=$W/runs/pat310_medium_pathfox_qwab$SUF/s$s
  local OUT=$W/hotpot_long/results_uniform/pat310_medium_pathfox_qwab$SUF/s$s/${tag}_ckpt15000/L$L
  mkdir -p "$OUT"
  { sbatch_head "FoxQ${SUF}_${tag}_s${s}_L${L}" "$gres" "$OUT/%j_eval.txt"; echo "$HEAD"
    echo "torchrun --nproc_per_node=$np --master_port=\${MASTER_PORT} ./run_clm.py --model_type gpt2 --tokenizer_name gpt2 --model_name_or_path $RUN/checkpoint-15000 --attn_implementation path_attn --bias_type wavelet --cfg_path $RUN/supply_model.cfg --dataset_name hotpot_qa --dataset_config_name distractor --hotpot_long_jsonl $W/hotpot_long/data/$jsonl --hotpot_long_lengths $L --do_eval --block_size $L --per_device_eval_batch_size $bs --path_attn_impl $impl --report_to none --output_dir $OUT --overwrite_output_dir --logging_dir $OUT/log --seed $s --path_use_qk_norm false --path_use_low_rank_w true --path_use_w_shortconv false --path_conv_size 3 --path_conv_bias false --num_harmonics 1 --single_A_B True --use_beta_modulation False --use_soft_wavelet_fox False --wavelet_baseline_use False --use_forget_gate True --qk_rotation False --ablate_switch False --wavelet_router False --load_best_model_at_end False"
  } > "$G/eval${SUF}_${tag}_s${s}_L${L}.sh"
}

for s in $SEEDS; do
  RUN=$W/runs/pat310_medium_pathfox_qwab$SUF/s$s
  mkdir -p "$RUN/train"
  { sbatch_head "FoxQ${SUF}_train_s$s" $TRAIN_GRES "$RUN/train/%j_train.txt"; echo "$HEAD"
    echo "cat > $RUN/supply_model.cfg <<'CFG'"
    sed -n "/<<'CFG'/,/^CFG\$/p" "$W/_gen_pat244_medium/train_mix_medium_K1_rho256_ricker_s42.sh" | sed '1d;$d'
    echo "forget_gate_bias_init=$B.0"
    echo 'CFG'
    echo "/cl/work5/hongyu-s/conda/envs/latest_transformers/bin/torchrun --nproc_per_node=2 --master_port=\"\${MASTER_PORT}\" ./run_clm.py --model_type gpt2 --tokenizer_name gpt2 --config_name openai-community/gpt2-medium --model_name_or_path runs/gpt2_medium_owt_pytorch_level_path_attn/checkpoint-80000 --dataset_name mix --block_size 512 --do_train --num_train_epochs 10 --logging_steps 500 --save_steps 5000 --save_only_model True --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --gradient_accumulation_steps 8 --learning_rate 1e-4 --weight_decay 0.0 --warmup_ratio 0.05 --attn_implementation path_attn --path_attn_impl pytorch --path_use_qk_norm false --path_use_low_rank_w true --path_use_w_shortconv false --path_conv_size 3 --path_conv_bias false --single_A_B True --share_freq_across_heads True --b_unfreeze_step 5000 --pe_method no_pe --num_harmonics 1 --wavelet_pe_softmax_use False --wavelet_mode logit_bias_ctxscale_shift_v0 --wavelet_baseline_use False --wavelet_router False --use_beta_modulation False --use_soft_wavelet_fox False --use_forget_gate True --full_fine_tune False --init_theta 0.847 --sample_num 16 --spectral_loss_coe 0.1 --temp_loss_coe 0.0 --distill_teacher wavelet --distill_in_which_layers 0 --distill_freq_scale 25 --smooth_use False --distilling_coe_warmup_use False --scale_range 0 16 --weight_alpha 0.0 --loss_type cos --qk_rotation False --router_band_num 8 --router_hidden_dim 32 --rel_selection all --bias_type wavelet --seed $s --overwrite_output_dir --output_dir $RUN --logging_dir $RUN/train_log --cfg_path $RUN/supply_model.cfg"
    echo "[ -d $RUN/checkpoint-15000 ] || { echo 'Missing final checkpoint' >&2; exit 1; }"
  } > "$G/train${SUF}_s$s.sh"
  gen_eval $s pytorch pytorch 512 2 2 p6000:2 hotpot_long_dev_uniform.jsonl
  gen_eval $s pytorch pytorch 2048 2 1 p6000:2 hotpot_long_dev_uniform.jsonl
  gen_eval $s pytorch pytorch 4096 4 1 p6000:4 hotpot_long_dev_uniform.jsonl
  gen_eval $s triton triton 4096 1 1 p6000:1 hotpot_long_dev_uniform.jsonl
  gen_eval $s triton triton 8192 1 1 p6000:1 hotpot_long_dev_uniform_8192only.jsonl
  gen_eval $s triton triton 12288 1 1 p6000:1 hotpot_long_dev_uniform_12288only.jsonl
  gen_eval $s triton triton 16384 1 1 p6000:1 hotpot_long_dev_uniform_16384_large_pool.jsonl
done

[ "$B" = 5 ] || exit 0  # submit.sh below is for the original b5 runs only
cat > "$G/submit.sh" <<'E'
#!/bin/bash
# bash -n every script, then submit each training with its evals chained afterok.
set -euo pipefail
G=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_pat310_pathfox_qwab
for f in "$G"/train_s*.sh "$G"/eval_*.sh; do bash -n "$f"; done
for s in 42 43 44; do
  T=$(sbatch --parsable "$G/train_s$s.sh"); echo "train s$s $T"
  for f in "$G"/eval_*_s${s}_L*.sh; do echo "  $(basename "$f") $(sbatch --parsable --dependency=afterok:$T "$f")"; done
done
E

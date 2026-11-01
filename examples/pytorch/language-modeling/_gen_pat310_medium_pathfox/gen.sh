#!/bin/bash
# PAT-310 W5: medium PaTH-FoX = paper medium PaTH pretrain (runs/gpt2_medium_owt_pytorch_level_path_attn/checkpoint-80000)
# + FoX forget gate (g_proj, --use_forget_gate True) added at finetune, paper protocol (evidence-first, no hotpot_respect_doc_order).
# Recipe = _gen_pat244_medium/train_mix_medium_PA_only_s43.sh (paper PaTH s43), at 2 GPU x 16 x accum 2 (global 64), except:
#   --use_forget_gate True, --path_attn_impl triton (forget gate only acts in the triton kernel), save every 1000 steps (model only).
# Variants: b0 = default init of the new g_proj (HF _init_weights: weight~N(0,0.02), bias 0 -> retention ~0.5/token, as FoX reference);
#           b5 = forget_gate_bias_init=5.0 (weight 0, bias 5 -> retention 0.9933/token, starts close to PaTH).
# Each train job is paired with a watcher eval job: every saved checkpoint, HotpotQA-Long L512/2048/4096, first 1000 examples, triton.
set -e
D=/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_pat310_medium_pathfox
LM=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
RES=$LM/hotpot_long/results_uniform/pat310_medium_pathfox
PATHFLAGS="--attn_implementation path_attn --path_use_qk_norm false --path_use_low_rank_w true --path_use_w_shortconv false --path_conv_size 3 --path_conv_bias false --single_A_B True --use_beta_modulation False --use_soft_wavelet_fox False --wavelet_baseline_use False --use_forget_gate True --qk_rotation False --wavelet_router False"
HDR='set -euxo pipefail
if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u; fi
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets WANDB_DISABLED=true WANDB_MODE=disabled
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID:-manual}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'
for V in b0 b5; do
for s in 42; do
N=pathfox_${V}_s$s; RUN=$LM/runs/pat310_medium_pathfox/$N
FGLINE=""; [ $V = b5 ] && FGLINE="forget_gate_bias_init=5.0"
cat > $D/train_$N.sh <<EOS
#!/bin/bash
#SBATCH --job-name=PAT310_medPathFoX_${V}_s$s
#SBATCH --output=$RUN/train/%j_train.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a100-80:2
#SBATCH --exclude=elm72,elm26
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
$HDR
cd $LM
RUN_OUT=$RUN
mkdir -p "\${RUN_OUT}/train"
MASTER_PORT=\$(( 23442 + SLURM_JOB_ID % 1000 ))
cat > "\${RUN_OUT}/supply_model.cfg" <<'CFG'
router_mode="seperate"
coe_mode="none"
tau=1
scale_type="none"
hotpot_question_position="later"
wavelet_ctxscale_k=1
$FGLINE
CFG
/cl/work5/hongyu-s/conda/envs/latest_transformers/bin/torchrun --nproc_per_node=2 --master_port="\${MASTER_PORT}" ./run_clm.py --model_type gpt2 --tokenizer_name gpt2 --config_name openai-community/gpt2-medium --model_name_or_path runs/gpt2_medium_owt_pytorch_level_path_attn/checkpoint-80000 --dataset_name mix --block_size 512 --do_train --num_train_epochs 10 --logging_steps 500 --save_steps 1000 --save_only_model True --per_device_train_batch_size 16 --per_device_eval_batch_size 4 --gradient_accumulation_steps 2 --learning_rate 1e-4 --weight_decay 0.0 --warmup_ratio 0.05 $PATHFLAGS --path_attn_impl triton --share_freq_across_heads True --b_unfreeze_step 5000 --pe_method vanilla --num_harmonics 1 --wavelet_pe_softmax_use False --wavelet_mode db1 --full_fine_tune False --init_theta 0.847 --sample_num 16 --spectral_loss_coe 0.1 --temp_loss_coe 0.0 --distill_teacher wavelet --distill_in_which_layers 0 --distill_freq_scale 25 --smooth_use False --distilling_coe_warmup_use False --scale_range 0 16 --weight_alpha 0.0 --loss_type cos --router_band_num 8 --router_hidden_dim 32 --rel_selection all --seed $s --overwrite_output_dir --output_dir "\${RUN_OUT}" --logging_dir "\${RUN_OUT}/train_log" --cfg_path "\${RUN_OUT}/supply_model.cfg"
EOS
cat > $D/evalwatch_$N.sh <<EOS
#!/bin/bash
#SBATCH --job-name=PAT310_medPathFoX_${V}_s${s}_ev
#SBATCH --output=$RES/logs/%j_${N}_evalwatch.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --exclude=elm72,elm26
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# Watcher: evaluates each checkpoint of $N as soon as it is fully written (paper protocol, first 1000 examples, triton).
$HDR
cd $LM
RUN=$RUN
for STEP in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000; do
  CK=\$RUN/checkpoint-\$STEP
  until [ -f \$CK/config.json ] && { [ -f \$CK/model.safetensors ] || [ -f \$CK/pytorch_model.bin ]; } && [ -f \$CK/trainer_state.json ]; do sleep 120; done
  sleep 60
  for L in 512 2048 4096; do
    OUT=$RES/$N/ckpt\${STEP}/L\${L}; [ -f \$OUT/eval_results.json ] && continue; mkdir -p \$OUT
    python ./run_clm.py --model_type gpt2 --tokenizer_name gpt2 --model_name_or_path \$CK --cfg_path \$RUN/supply_model.cfg --dataset_name hotpot_qa --dataset_config_name distractor --hotpot_long_jsonl $LM/hotpot_long/data/hotpot_long_dev_uniform.jsonl --hotpot_long_lengths \$L --do_eval --block_size \$L --per_device_eval_batch_size 1 --max_eval_samples 1000 --path_attn_impl triton --report_to none --output_dir \$OUT --overwrite_output_dir --logging_dir \$OUT/log --seed $s $PATHFLAGS --path_conv_bias false --num_harmonics 1 --ablate_switch False --load_best_model_at_end False
  done
done
EOS
mkdir -p $RUN/train $RES/logs
done; done

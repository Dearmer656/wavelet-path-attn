#!/bin/bash
#SBATCH --job-name=PAT310_pathfox_docorder_s42
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat310_fox_docorder/small_pathfox_docorder_s42/train/%j_train.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:p6000:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=100:00:00
#SBATCH --exclude=elm72
# PAT-310 W5: PaTH-FoX baseline. Finetune the from-scratch PaTH-FoX pretrain (runs/1r_FoX_baseline_from_s, same pretrain schedule as the PaTH backbone; final step 107550 vs PaTH ckpt-80000) with the identical finetune recipe. Forget gate only acts in the triton kernel, so path_attn_impl=triton for train and eval.
# (Qiu et al. 2025, elementwise variant; fla commit 0f6bd1d6db), added at finetune.
# Recipe identical to the canonical small QWAB K1 rho=256 Ricker finetune
# (_gen_ricker_backfill/train_K1_L512_me16_rho256_ricker_s42.sh) and the PaTH baseline
# (PA_baseline_multi_seeds): backbone (here: runs/1r_FoX_baseline_from_s), mix
# dataset, block 512, lr 1e-4, wd 0, warmup 0.05 linear, 10 epochs, fp32,
# global batch 64 (here 2 GPU x 32, accum 1, vs 4 x 16 there), eval at checkpoint-15000.
set -euxo pipefail
if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u; fi
cd "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling"
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID:-manual}
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true WANDB_MODE=disabled
RUN_OUT="/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat310_fox_docorder/small_pathfox_docorder_s42"
MASTER_PORT=$((12000 + SLURM_JOB_ID % 20000))
cp /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat310_fox_docorder/supply_model_fox_docorder.cfg "${RUN_OUT}/supply_model.cfg"
python -m torch.distributed.run --nproc_per_node=2 --master_port="${MASTER_PORT}" ./run_clm.py --model_type gpt2 --tokenizer_name gpt2 --config_name gpt2 --share_freq_across_heads True --learning_rate 1e-4 --weight_decay 0.0 --per_device_train_batch_size 32 --per_device_eval_batch_size 16 --block_size 512 --dataset_name mix --do_train --eval_strategy no --logging_dir "${RUN_OUT}/train_log" --logging_steps 500 --num_train_epochs 10 --num_harmonics 1 --wavelet_pe_softmax_use False --save_steps 2500 --attn_implementation path_attn --path_use_qk_norm false --path_use_low_rank_w true --path_use_w_shortconv false --path_conv_size 3 --warmup_ratio 0.05 --path_conv_bias false --output_dir "${RUN_OUT}" --overwrite_output_dir --gradient_accumulation_steps 1 --b_unfreeze_step 5000 --pe_method no_pe --single_A_B True --use_beta_modulation False --use_soft_wavelet_fox False --wavelet_mode off --model_name_or_path runs/1r_FoX_baseline_from_s --full_fine_tune False --wavelet_baseline_use False --init_theta 0.847 --use_forget_gate True --path_attn_impl triton --sample_num 16 --spectral_loss_coe 0.1 --temp_loss_coe 0.0 --distill_teacher wavelet --distill_in_which_layers 0 --distill_freq_scale 25 --smooth_use False --distilling_coe_warmup_use False --scale_range 0 16 --weight_alpha 0.0 --loss_type cos --qk_rotation False --wavelet_router False --router_band_num 8 --router_hidden_dim 32 --rel_selection all --cfg_path "${RUN_OUT}/supply_model.cfg" --seed 42
[ -d "${RUN_OUT}/checkpoint-15000" ] || { echo "Missing final checkpoint" >&2; exit 1; }
for L in 512 2048 4096; do sbatch "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_pat310_fox_docorder/eval_hp_pathfox_docorder_s42_L${L}.sh"; done
sbatch "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_pat310_fox_docorder/eval_xsum_pathfox_docorder_s42.sh"

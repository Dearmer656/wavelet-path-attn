#!/bin/bash
# PAT-310 Table 1 backfill, step 1: current-code reproduction of the s42 rows before training s43/s44.
# (a) re-evaluate the s42 checkpoints behind Table 1 with exactly the original eval args (same file, GPUs, batch size);
# (b) re-run the s42 fine-tune with the original cfg for ~1000 steps and compare the logged loss at steps 500/1000.
set -eu
D=/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_pat310_table1
W=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
HDR='set -euxo pipefail
set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets WANDB_DISABLED=true WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd '"$W"
WAV='--wavelet_router False --router_band_num 8 --scale_range 0 16 --wavelet_mode logit_bias_ctxscale_shift_v0 --wavelet_baseline_use False --use_beta_modulation False --use_soft_wavelet_fox False --single_A_B True --num_harmonics 1 --share_freq_across_heads True'
# name ckpt_run L ngpu bs jsonl pe_args expected
mk_eval() {
  local name=$1 run=$2 L=$3 ng=$4 bs=$5 js=$6 pe=$7 exp=$8
  local out=$W/hotpot_long/results_pat310_table1_repro/$name/L$L f=$D/repro_eval_${name}_L$L.sh
  cat > $f <<EOS
#!/bin/bash
#SBATCH --job-name=T1rep_${name}_L$L
#SBATCH --output=$out/%j_eval.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:p6000:$ng
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# PAT-310 Table 1: current-code re-eval of $run/checkpoint-15000 at L$L with the original eval args ($ng GPUs, bs $bs, $(basename $js)). Paper value: F1 $exp.
$HDR
MASTER_PORT=\$(( 20000 + SLURM_JOB_ID % 10000 ))
/cl/work5/hongyu-s/conda/envs/latest_transformers/bin/torchrun --nproc_per_node=$ng --master_port="\${MASTER_PORT}" ./run_clm.py --model_type gpt2 --tokenizer_name gpt2 --model_name_or_path $W/runs/$run/checkpoint-15000 $pe --dataset_name hotpot_qa --dataset_config_name distractor --hotpot_long_jsonl $W/hotpot_long/data/$js --hotpot_long_lengths $L --do_eval --block_size $L --per_device_eval_batch_size $bs --output_dir $out --overwrite_output_dir --logging_dir $out/log --load_best_model_at_end False --seed 42
python3 -c "import json; d=json.load(open('$out/eval_results.json')); print('REPRO $name L$L F1=%.2f EM=%.2f n=%d (paper F1 $exp)' % (100*d['eval_f1'], 100*d['eval_em'], d['eval_samples']))"
EOS
  mkdir -p $out
}
mk_eval alibi_s42 mix_medium_owt_alibi_10ep_s42_fp32 512 2 1 hotpot_long_dev.jsonl "--attn_implementation eager --pe_method alibi" 81.02
mk_eval alibi_s42 mix_medium_owt_alibi_10ep_s42_fp32 4096 2 1 hotpot_long_dev.jsonl "--attn_implementation eager --pe_method alibi" 20.71
mk_eval rotary_s42 mix_medium_owt_rotary_10ep_s42_fp32 512 2 1 hotpot_long_dev.jsonl "--attn_implementation eager --pe_method rotary" 81.16
mk_eval ntk_s42 mix_medium_owt_rotary_10ep_s42_fp32 4096 3 4 hotpot_long_dev.jsonl "--attn_implementation eager --pe_method rotary --rope_theta 85550.3759" 31.48
mk_eval nope_s42 mix_medium_owt_nope_10ep 512 4 4 hotpot_long_dev_uniform.jsonl "--pe_method no_pe --attn_implementation eager $WAV" 80.59
mk_eval nope_s42 mix_medium_owt_nope_10ep 4096 4 1 hotpot_long_dev_uniform.jsonl "--pe_method no_pe --attn_implementation eager $WAV" 75.42
# name pretrain pdbs accum(2 GPUs) attn pe_args loss500 loss1000 config_name_arg (as in the original script)
mk_train() {
  local name=$1 pre=$2 pb=$3 ac=$4 attn=$5 pe=$6 l5=$7 l10=$8 cfgn=$9
  local out=$W/runs/pat310_table1_repro/${name}_s42 f=$D/repro_train_${name}_s42.sh
  cat > $f <<EOS
#!/bin/bash
#SBATCH --job-name=T1rep_train_${name}
#SBATCH --output=$out/%j_train.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:p6000:2
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# PAT-310 Table 1: current-code re-run of the s42 $name medium fine-tune, original cfg (per-device bs $pb; 2 GPUs x accum $ac = global 64,
# same per-step sample set as the original 4-GPU run), stopped by timeout after ~1000 steps. Original s42 loss: step 500 $l5, step 1000 $l10.
$HDR
MASTER_PORT=\$(( 21000 + SLURM_JOB_ID % 10000 ))
timeout 75m /cl/work5/hongyu-s/conda/envs/latest_transformers/bin/torchrun --nproc_per_node=2 --master_port="\${MASTER_PORT}" ./run_clm.py --model_type gpt2 --tokenizer_name gpt2 $cfgn --model_name_or_path $W/runs/$pre/checkpoint-80000 --dataset_name mix --block_size 512 --do_train --num_train_epochs 10 --logging_steps 500 --save_steps 100000 --per_device_train_batch_size $pb --per_device_eval_batch_size $pb --gradient_accumulation_steps $ac --learning_rate 1e-4 --weight_decay 0.0 --warmup_ratio 0.05 --attn_implementation $attn $pe --preprocessing_num_workers 8 --ddp_timeout 21600 --seed 42 --overwrite_output_dir --output_dir $out --logging_dir $out/tensorboard || echo "stopped (timeout expected)"
grep -o "{'loss': [0-9.]*[^}]*'epoch': [0-9.]*}" $out/*_train.txt | head -3; echo "original s42: step500 $l5 step1000 $l10"
EOS
  mkdir -p $out
}
mk_train alibi gpt2_medium_owt_alibi_flash_80k_a6000x4 8 4 flash_attention_2 "--pe_method alibi $WAV" 1.7003 1.6266 ""
mk_train rotary gpt2_medium_owt_rotary_80k 2 16 eager "--pe_method rotary $WAV" 1.7296 1.6519 ""
mk_train nope gpt2_medium_owt_nope_80k 4 8 eager "--pe_method no_pe $WAV" 1.7093 1.6533 "--config_name openai-community/gpt2-medium"

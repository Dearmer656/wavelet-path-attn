#!/bin/bash
# PAT-310 Table 1 backfill, step 2: s43/s44 fine-tunes of the medium ALiBi / RoPE / NoPE rows (RoPE-NTK = eval-time theta on the RoPE
# checkpoints), each an exact copy of the s42 script's cfg (checked against the saved training_args) with only seed/output changed,
# on 2 GPUs with gradient accumulation doubled (global batch 64, same per-device batch). Evals copy each row's s42 eval args
# (same data file, GPU count and batch size); the only change is L16384, which uses the full 7403-example large_pool file
# (the s42 ALiBi/RoPE/NoPE L16384 numbers came from a 69-example file), so s42 is re-evaluated there as well.
set -eu
D=/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_pat310_table1
W=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
HDR='set -euxo pipefail
set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets WANDB_DISABLED=true WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SKIP_FENICE=1 SKIP_SUMMAC=1
cd '"$W"
WAV='--wavelet_router False --router_band_num 8 --scale_range 0 16 --wavelet_mode logit_bias_ctxscale_shift_v0 --wavelet_baseline_use False --use_beta_modulation False --use_soft_wavelet_fox False --single_A_B True --num_harmonics 1 --share_freq_across_heads True'
GPU=${GPU:-p6000}
declare -A RUN PRE PB AC ATTN CFGN EPE FRONT
RUN[alibi]=mix_medium_owt_alibi_10ep_s@_fp32; PRE[alibi]=gpt2_medium_owt_alibi_flash_80k_a6000x4; PB[alibi]=8; AC[alibi]=4; ATTN[alibi]=flash_attention_2; CFGN[alibi]=""; EPE[alibi]="--attn_implementation eager --pe_method alibi"; FRONT[alibi]=hotpot_long_dev.jsonl
RUN[rotary]=mix_medium_owt_rotary_10ep_s@_fp32; PRE[rotary]=gpt2_medium_owt_rotary_80k; PB[rotary]=2; AC[rotary]=16; ATTN[rotary]=eager; CFGN[rotary]=""; EPE[rotary]="--attn_implementation eager --pe_method rotary"; FRONT[rotary]=hotpot_long_dev.jsonl
RUN[nope]=mix_medium_owt_nope_10ep_s@; PRE[nope]=gpt2_medium_owt_nope_80k; PB[nope]=4; AC[nope]=8; ATTN[nope]=eager; CFGN[nope]="--config_name openai-community/gpt2-medium"; EPE[nope]="--pe_method no_pe --attn_implementation eager $WAV"; FRONT[nope]=hotpot_long_dev_uniform.jsonl
declare -A NTK=( [2048]=41829.37 [4096]=85550.3759 [8192]=174969.5854 [12288]=265909.7030 [16384]=357851.7978 )
runname() { local m=$1 s=$2; if [ $m = nope ] && [ $s = 42 ]; then echo mix_medium_owt_nope_10ep; else echo ${RUN[$m]/@/$s}; fi; }
jsonl() { local m=$1 L=$2; if [ $L -le 4096 ]; then echo ${FRONT[$m]}; elif [ $L -eq 16384 ]; then echo hotpot_long_dev_uniform_16384_large_pool.jsonl; else echo hotpot_long_dev_uniform_${L}only.jsonl; fi; }
# tag model seed L ngpu bs extra_pe
mk_hp() {
  local tag=$1 m=$2 s=$3 L=$4 ng=$5 bs=$6 xpe=$7 run; run=$(runname $m $s)
  local out=$W/hotpot_long/results_pat310_table1/${tag}_s$s/L$L
  cat > $D/eval_hp_${tag}_s${s}_L$L.sh <<EOS
#!/bin/bash
#SBATCH --job-name=T1_hp_${tag}_s${s}_L$L
#SBATCH --output=$out/%j_eval.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:$GPU:$ng
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# PAT-310 Table 1: $tag medium s$s HotpotQA-Long L$L, args of the s42 Table 1 eval ($ng GPUs, bs $bs, $(jsonl $m $L)), full set, checkpoint-15000.
$HDR
CKPT=$W/runs/$run/checkpoint-15000
[ -d "\${CKPT}" ] || { echo "Missing \${CKPT}" >&2; exit 1; }
MASTER_PORT=\$(( 20000 + SLURM_JOB_ID % 10000 ))
/cl/work5/hongyu-s/conda/envs/latest_transformers/bin/torchrun --nproc_per_node=$ng --master_port="\${MASTER_PORT}" ./run_clm.py --model_type gpt2 --tokenizer_name gpt2 --model_name_or_path "\${CKPT}" ${EPE[$m]} $xpe --dataset_name hotpot_qa --dataset_config_name distractor --hotpot_long_jsonl $W/hotpot_long/data/$(jsonl $m $L) --hotpot_long_lengths $L --do_eval --block_size $L --per_device_eval_batch_size $bs --output_dir $out --overwrite_output_dir --logging_dir $out/log --load_best_model_at_end False --seed 42
python3 -c "import json; d=json.load(open('$out/eval_results.json')); print('$tag s$s L$L F1=%.2f EM=%.2f n=%d' % (100*d['eval_f1'], 100*d['eval_em'], d['eval_samples']))"
EOS
  mkdir -p $out
}
# tag model seed ngpu thetas(3 or empty)
mk_xsum() {
  local tag=$1 m=$2 s=$3 ng=$4 th=$5 run; run=$(runname $m $s)
  local rd=$W/runs/$run/ckpt_eval_xsum${th:+_ntk}
  cat > $D/eval_xsum_${tag}_s$s.sh <<EOS
#!/bin/bash
#SBATCH --job-name=T1_xsum_${tag}_s$s
#SBATCH --output=$rd/%j_xsum_${tag}_medium_mix_10ep.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:$GPU:$ng
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# PAT-310 Table 1: $tag medium s$s Filtered XSum, args of the s42 Table 1 eval ($ng GPUs, bs 16/8/4 at L512/1024/1536${th:+, NTK theta $th}), checkpoint-15000.
$HDR
CKPT=$W/runs/$run/checkpoint-15000
[ -d "\${CKPT}" ] || { echo "Missing \${CKPT}" >&2; exit 1; }
CSV=$rd/xsum_filter_metrics_${tag}_medium_mix_10ep_\${SLURM_JOB_ID}.csv
echo "block_size,batch_size,theta,rouge1,rouge2,rougeL,count" > "\${CSV}"
BS=(512 1024 1536); BAT=(16 8 4); TH=(${th:-0 0 0})
for i in 0 1 2; do
  OUT=$rd/xsum_out_\${SLURM_JOB_ID}/bs_\${BS[\$i]}; mkdir -p "\${OUT}"
  if [ "\${TH[\$i]}" = 0 ]; then XT=""; else XT="--rope_theta \${TH[\$i]}"; fi
  /cl/work5/hongyu-s/conda/envs/latest_transformers/bin/torchrun --nproc_per_node=$ng --master_port=\$(( 19800 + SLURM_JOB_ID % 1000 )) ./run_clm.py --model_type gpt2 --tokenizer_name gpt2 --model_name_or_path "\${CKPT}" ${EPE[$m]} $( [ $m = nope ] || echo "$WAV" ) \${XT} --block_size \${BS[\$i]} --dataset_name xsum --dataset_config_name default --validation_file /cl/work5/hongyu-s/fact-check-summarization/xsum_test_filter_level2_official_style.jsonl --do_eval --per_device_eval_batch_size \${BAT[\$i]} --output_dir "\${OUT}" --overwrite_output_dir --xsum_bucket_size 512 --xsum_bucket_apply_to eval_test --load_best_model_at_end False --seed 42
  python3 -c "import json; d=json.load(open('\${OUT}/eval_results.json')); print('\${BS[\$i]},\${BAT[\$i]},\${TH[\$i]},%s,%s,%s,%s' % tuple(d.get(k) for k in ('eval_rouge1','eval_rouge2','eval_rougeL','eval_count')))" >> "\${CSV}"
done
cat "\${CSV}"
EOS
  mkdir -p $rd
}
mk_train() {
  local m=$1 s=$2 run; run=$(runname $m $s)
  local out=$W/runs/$run
  cat > $D/train_${m}_s$s.sh <<EOS
#!/bin/bash
#SBATCH --job-name=T1_train_${m}_s$s
#SBATCH --output=$W/log_file/train/%j_${run}.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:$GPU:2
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# PAT-310 Table 1: $m medium mix fine-tune, seed $s. cfg = the s42 run's saved training_args (per-device bs ${PB[$m]}, attn ${ATTN[$m]},
# lr 1e-4, wd 0, warmup_ratio 0.05, 10 epochs, linear, fp32); 2 GPUs x accum ${AC[$m]} = global batch 64 (s42 used 4 GPUs x accum $(( ${AC[$m]} / 2 ))).
$HDR
MASTER_PORT=\$(( 21000 + SLURM_JOB_ID % 10000 ))
/cl/work5/hongyu-s/conda/envs/latest_transformers/bin/torchrun --nproc_per_node=2 --master_port="\${MASTER_PORT}" ./run_clm.py --model_type gpt2 --tokenizer_name gpt2 ${CFGN[$m]} --model_name_or_path $W/runs/${PRE[$m]}/checkpoint-80000 --dataset_name mix --block_size 512 --do_train --num_train_epochs 10 --logging_steps 500 --save_steps 5000 --per_device_train_batch_size ${PB[$m]} --per_device_eval_batch_size ${PB[$m]} --gradient_accumulation_steps ${AC[$m]} --learning_rate 1e-4 --weight_decay 0.0 --warmup_ratio 0.05 --attn_implementation ${ATTN[$m]} --pe_method $( [ $m = nope ] && echo no_pe || echo $m ) $WAV --preprocessing_num_workers 8 --ddp_timeout 21600 --seed $s --overwrite_output_dir --output_dir $out --logging_dir $out/tensorboard
EOS
}
for s in 43 44; do
  for m in alibi rotary nope; do mk_train $m $s; done
  for L in 512 2048 4096 8192 12288 16384; do mk_hp alibi alibi $s $L 2 1 ""; mk_hp rotary rotary $s $L 2 1 ""; done
  for L in 2048 4096 8192 12288 16384; do mk_hp ntk rotary $s $L 3 4 "--rope_theta ${NTK[$L]}"; done
  mk_hp nope nope $s 512 4 4 ""; mk_hp nope nope $s 2048 4 2 ""; for L in 4096 8192 12288 16384; do mk_hp nope nope $s $L 4 1 ""; done
  mk_xsum alibi alibi $s 4 ""; mk_xsum rotary rotary $s 4 ""; mk_xsum nope nope $s 2 ""; mk_xsum ntk rotary $s 2 "10000.0000 20452.2287 31082.2367"
done
mk_hp alibi alibi 42 16384 2 1 ""; mk_hp rotary rotary 42 16384 2 1 ""; mk_hp nope nope 42 16384 4 1 ""

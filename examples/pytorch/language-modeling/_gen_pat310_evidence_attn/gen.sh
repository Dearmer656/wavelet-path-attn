#!/bin/bash
# PAT-310 W5: attention mass on supporting facts at the answer position, medium FoX (s42-44) / PaTH (s43, s44) / QWAB rho256 (s42-44),
# doc-order evidence (hotpot_respect_doc_order=True), HotpotQA-Long L512/2048/4096, first N eval examples, bs 1, 1 GPU.
# Usage: bash gen.sh N [smoke]   -> writes probe_<name>.sh
set -e
N=${1:-300}; SMOKE=${2:-}
D=/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_pat310_evidence_attn
LM=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
RES=$LM/hotpot_long/results_uniform/pat310_evidence_attn
JSONL=$LM/hotpot_long/data/hotpot_long_dev_uniform.jsonl
LENS="512 2048 4096"; [ -n "$SMOKE" ] && LENS="512"
COMMON="--tokenizer_name gpt2 --dataset_name hotpot_qa --dataset_config_name distractor --hotpot_long_jsonl $JSONL --do_eval --per_device_eval_batch_size 1 --report_to none --overwrite_output_dir --load_best_model_at_end False --max_eval_samples $N"
PATHFLAGS="--model_type gpt2 --attn_implementation path_attn --path_attn_impl pytorch --path_use_qk_norm false --path_use_low_rank_w true --path_use_w_shortconv false --path_conv_size 3 --path_conv_bias false --num_harmonics 1 --single_A_B True --use_beta_modulation False --use_soft_wavelet_fox False --wavelet_baseline_use False --use_forget_gate False --qk_rotation False --ablate_switch False --wavelet_router False"
mk () { # name kind seed ckpt cfg extra
  local NAME=$1 KIND=$2 S=$3 CK=$4 CFG=$5 EXTRA=$6
  cat > $D/probe_${NAME}${SMOKE:+_smoke}.sh <<EOS
#!/bin/bash
#SBATCH --job-name=PAT310_evattn_${NAME}${SMOKE:+_smoke}
#SBATCH --output=$RES/logs/%j_${NAME}${SMOKE:+_smoke}.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --exclude=elm72,elm26
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
set -eo pipefail
source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:\${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets WANDB_DISABLED=true WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $LM
for L in $LENS; do
  OUT=$RES/${NAME}${SMOKE:+_smoke}/L\${L}; mkdir -p \$OUT
  [ -f \$OUT/evidence_attn.json ] && continue
  python $D/probe_evidence_attention.py --probe_out \$OUT/evidence_attn.json --probe_n $N --probe_model $KIND -- --model_name_or_path $CK --cfg_path $CFG $EXTRA $COMMON --hotpot_long_lengths \$L --block_size \$L --output_dir \$OUT --logging_dir \$OUT/log --seed $S
done
EOS
}
for s in 42 43 44; do
  mk FoX_s$s fox $s $LM/runs/pat310_fox_docorder/medium_fox_docorder_s$s/checkpoint-15000 $LM/_gen_pat310_fox_medium_docorder/cfgs/docorder.cfg "--model_type forgetting_transformer"
done
for s in 43 44; do
  C=$D/cfgs/PaTH_s$s.cfg; mkdir -p $D/cfgs; cp $LM/runs/mix_medium_PA_only_s$s/supply_model.cfg $C; echo 'hotpot_respect_doc_order=True' >> $C
  mk PaTH_s$s path $s $LM/runs/mix_medium_PA_only_s$s/checkpoint-15000 $C "$PATHFLAGS"
done
for s in 42 43 44; do
  C=$D/cfgs/QWAB_s$s.cfg; mkdir -p $D/cfgs; cp $LM/runs/mix_medium_K1_rho256_ricker_s$s/supply_model.cfg $C; echo 'hotpot_respect_doc_order=True' >> $C
  BT=$(grep -oP '(?<=bias_type=")[^"]+' $C)
  mk QWAB_s$s path $s $LM/runs/mix_medium_K1_rho256_ricker_s$s/checkpoint-15000 $C "$PATHFLAGS --bias_type $BT"
done
mkdir -p $RES/logs

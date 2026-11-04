#!/bin/bash
# Submit one model's fine-tune for one seed and chain its evals with afterok. Usage: bash submit_seed.sh <alibi|rotary|nope> <43|44>
set -eu
cd /project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_pat310_table1
m=$1; s=$2
for f in train_${m}_s$s.sh eval_*_${m}_s${s}*.sh $( [ $m = rotary ] && ls eval_*_ntk_s${s}*.sh ); do bash -n $f; done
J=$(sbatch --parsable --exclude=elm72 train_${m}_s$s.sh); echo "train $m s$s: $J"
for f in eval_hp_${m}_s${s}_L*.sh eval_xsum_${m}_s$s.sh $( [ $m = rotary ] && ls eval_hp_ntk_s${s}_L*.sh eval_xsum_ntk_s$s.sh ); do echo "  $f: $(sbatch --parsable --exclude=elm72 --dependency=afterok:$J $f)"; done

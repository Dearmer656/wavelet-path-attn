#!/bin/bash
# bash -n every script, then submit each training with its evals chained afterok.
set -euo pipefail
G=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_pat310_pathfox_qwab
for f in "$G"/train_s*.sh "$G"/eval_*.sh; do bash -n "$f"; done
for s in 42 43 44; do
  T=$(sbatch --parsable "$G/train_s$s.sh"); echo "train s$s $T"
  for f in "$G"/eval_*_s${s}_L*.sh; do echo "  $(basename "$f") $(sbatch --parsable --dependency=afterok:$T "$f")"; done
done

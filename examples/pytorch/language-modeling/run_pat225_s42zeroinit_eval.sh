#!/bin/bash
set -euo pipefail
LANG_DIR=/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling
cd "${LANG_DIR}/hotpot_long"
CKPT="${LANG_DIR}/runs/pat225_scale_card/S4_s42_zeroinit/checkpoint-15000"
CFG="${LANG_DIR}/runs/pat225_scale_card/S4_s42_zeroinit/supply_model.cfg"
# 4096 first per request, then 2048/512. Pinned to elm61 (training node, a6000).
for L in 4096 2048 512; do
  sbatch --gres=gpu:a6000:4 --nodelist=elm61 --dependency=afterok:518115 \
    --output="${LANG_DIR}/hotpot_long/logs/%j_s42zeroinit_L${L}.txt" \
    run_eval_hotpot_long_uniform_a6000.sh "${CKPT}" pat225_S4_s42_zeroinit_ckpt15000 "${L}" "${CFG}" "${L}" pytorch
done
echo "submitted s42_zeroinit eval L4096/L2048/L512 (dependency on 518115)"

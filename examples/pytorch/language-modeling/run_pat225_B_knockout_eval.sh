#!/bin/bash
set -euo pipefail
# PAT-225 Direction B: QWAB-off (all-layer wavelet knockout) eval on trained
# K4 checkpoints. QWAB-on F1 already exists (s42/s43/s44 L4096 = .7038/.6446/.6477),
# so we only run OFF here; C_S = on - off. If C_S ~ 0 everywhere, QWAB's benefit
# is baked into the backbone during training, not an active inference-time bias.
LANG_DIR=/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling
cd "${LANG_DIR}/hotpot_long"
SC="${LANG_DIR}/runs/pat225_scale_card"
# L4096 QWAB-off for all 3 seeds (decisive length)
for s in s42 s43 s44; do
  CKPT="${SC}/S4_${s}/checkpoint-15000"
  OFFCFG="${SC}/S4_${s}/supply_model_qwaboff.cfg"
  sbatch --gres=gpu:a6000:4 \
    --output="${LANG_DIR}/hotpot_long/logs/%j_B_qwaboff_${s}_L4096.txt" \
    run_eval_hotpot_long_uniform_a6000.sh "${CKPT}" "pat225_B_qwaboff_${s}" 4096 "${OFFCFG}" 4096 pytorch
done
# L512 in-range control (s42 only): expect C_S ~ 0 here regardless
sbatch --gres=gpu:a6000:4 \
  --output="${LANG_DIR}/hotpot_long/logs/%j_B_qwaboff_s42_L512.txt" \
  run_eval_hotpot_long_uniform_a6000.sh "${SC}/S4_s42/checkpoint-15000" "pat225_B_qwaboff_s42" 512 "${SC}/S4_s42/supply_model_qwaboff.cfg" 512 pytorch
echo "submitted B knockout evals: L4096 off x3 seeds + L512 off s42"

#!/bin/bash
#SBATCH --job-name=motif_delta
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_motif_delta.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:1
#SBATCH --time=0:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
# Final aggregation step: reads the QWAB and PA-only motif_retention_summary_*.json
# files and writes delta_motif_summary.md. No GPU work, just routed through a
# compute node per project policy (this is trivial, dependency-chained after
# both eval jobs).

_slack() {
    python3 /project/nlp-work5/hongyu-s/gate1/scripts/notify_slack.py \
        --exit-code "$1" \
        --job-id "${SLURM_JOB_ID}" \
        --node "${SLURMD_NODENAME}" \
        --issue "PAT-194" \
        --gpu "6000x1" \
        --summary "PAT-194 Delta_motif(L) = G_QWAB - G_PAonly final aggregation"
}
trap '_slack $?' EXIT

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi

OUT_DIR=/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/analysis_outputs/motif_retention_gap
cd /project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long

python compute_delta_motif.py \
  --qwab_summary "${OUT_DIR}/motif_retention_summary_qwab.json" \
  --paonly_summary "${OUT_DIR}/motif_retention_summary_paonly.json" \
  --out_dir "${OUT_DIR}"

echo "=== DONE: Delta_motif(L) computed ==="
cat "${OUT_DIR}/delta_motif_summary.md"

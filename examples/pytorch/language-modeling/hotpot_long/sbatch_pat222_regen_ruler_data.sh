#!/bin/bash
#SBATCH --job-name=pat222_regen_ruler
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_pat222_regen_ruler.txt
#SBATCH --partition=lang_long
#SBATCH --account=lang
#SBATCH --nodelist=ahcclcsa01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00

# PAT-222: Regenerate RULER eval data (L=2048/4096/8192) after padding bug fix.
# Bug: fine-grained Pad words were inserted AFTER qtail (question); now fixed to go before.

set -euxo pipefail
if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
    set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi

BASE=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long
cd "${BASE}"
mkdir -p logs data

python3 build_ruler_len_splits.py \
    --infile "${BASE}/data/ruler_smoke_100.jsonl" \
    --tokenizer_json /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/ruler_ft_matrix_20260525/path_pa_s42_save1k/tokenizer.json \
    --outdir "${BASE}/data" \
    --train_count 2000 \
    --eval_count 200

echo "=== Done: RULER data regenerated (2048/4096/8192) ==="

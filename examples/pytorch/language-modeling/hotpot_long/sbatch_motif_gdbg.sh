#!/bin/bash
#SBATCH --job-name=motif_gdbg
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_motif_gdbg.txt
#SBATCH --partition=gpu_short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:3090:1
#SBATCH --nodelist=elm54
#SBATCH --time=0:30:00
set -euxo pipefail
if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then set +u; . /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u; fi
export PYTHONPATH=/cl/work5/hongyu-s/flash-linear-attention:/cl/work5/hongyu-s/transformers/src${PYTHONPATH:+:${PYTHONPATH}}
export HF_HOME=/cl/work5/hongyu-s/huggingfac; export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true; export PYTHONUNBUFFERED=1
BASE=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
cd "${BASE}/hotpot_long"; OUT=${BASE}/hotpot_long/analysis_outputs/distill_bias
python motif_gen_debug.py --checkpoint "${BASE}/runs/pat217_motif_ft/motif_real_lam1.0_k16/checkpoint-3000" --jsonl "${BASE}/hotpot_long/data/hotpot_long_dev_uniform.jsonl" --npz "${OUT}/distilled_bias_L512.npz" --recon_csv "${OUT}/recon_error_L2048.csv" --L 2048

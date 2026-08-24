#!/bin/bash
#SBATCH --job-name=g0_K3_learned_s43
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/analysis/gate_usage/logs/%j_K3_learned_s43.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
set -euxo pipefail
if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u; fi
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export WANDB_DISABLED=true WANDB_MODE=disabled
cd /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
python analysis/gate_usage/dump_gate_by_position.py \
  --checkpoint "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat244_dual_temp/K3_L512_learnedRatioRmsBoth_128_256_384_s43/checkpoint-15000" \
  --cfg_path "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat244_dual_temp/K3_L512_learnedRatioRmsBoth_128_256_384_s43/supply_model.cfg" \
  --block_size 512 --n_batches 20 --batch_size 8 --seed 43 \
  --out_path analysis/gate_usage/results/K3_learnedRatioRmsBoth_s43.json

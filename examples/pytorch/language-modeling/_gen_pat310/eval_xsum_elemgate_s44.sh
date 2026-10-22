#!/bin/bash
#SBATCH --job-name=PAT310_xsumGate_s44
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/log_file/train/%j_PAT310_xsumGate_s44.txt
#SBATCH --partition=gpu_long
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:a6000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
set -uxo pipefail
# PAT-310 E1d (SDPA output gate, wavelet off) small XSum, paper protocol (same helper as Table 4 basis runs)
bash /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_xsum_basis/eval_filter_xsum_bias_fixed_1gpu.sh /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat310_sdpa_gate/small_elemgate_s44 off 44 small_elemgate_s44

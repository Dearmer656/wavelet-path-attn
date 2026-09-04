#!/bin/bash
#SBATCH --job-name=build_flash_attn
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_build_flash_attn.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Compile flash-attn from source into the latest_transformers conda env, using the
# conda-installed cuda-nvcc toolchain (CUDA_HOME = the env prefix itself, since
# conda install -c nvidia cuda-nvcc puts nvcc directly under <env>/bin).

set -euxo pipefail
if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi

export CUDA_HOME="${CONDA_PREFIX}"
echo "CUDA_HOME=${CUDA_HOME}"
nvcc --version

export MAX_JOBS=8
pip install flash-attn --no-build-isolation -v 2>&1 | tail -200

python3 -c "import flash_attn; print('flash_attn version:', flash_attn.__version__)"

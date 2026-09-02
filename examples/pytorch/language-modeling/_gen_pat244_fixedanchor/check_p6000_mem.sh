#!/bin/bash
#SBATCH --job-name=check_p6000_mem
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/check_p6000_mem_output.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:p6000:1
#SBATCH --nodelist=elm82
#SBATCH --time=00:02:00
#SBATCH --ntasks=1
nvidia-smi --query-gpu=name,memory.total --format=csv

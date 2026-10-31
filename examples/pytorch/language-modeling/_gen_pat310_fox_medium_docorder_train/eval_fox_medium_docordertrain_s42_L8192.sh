#!/bin/bash
#SBATCH --job-name=PAT310_foxmedDT_L8192
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/logs/%j_fox_medium_docordertrain_s42_L8192.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:2
#SBATCH --exclude=elm72
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# PAT-310: medium FoX finetuned with evidence in document order (runs/pat310_fox_docorder/medium_fox_docorder_s42 ckpt-15000) HotpotQA-Long L8192 with the
# evidence kept at its original document position (hotpot_respect_doc_order=True), same jsonl and condition as the paper's
# medium PaTH s43/s44 + QWAB long-length evals (08-26..08-30 doc-order window). Otherwise identical to eval_fox_medium_mix_s42_hotpot_alllen.sh.
set -euxo pipefail
set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
CKPT=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat310_fox_docorder/medium_fox_docorder_s42/checkpoint-15000
[ -d "${CKPT}" ] || { echo "Missing ${CKPT}" >&2; exit 1; }
MASTER_PORT=$(( 17500 + SLURM_JOB_ID % 10000 ))
python -m torch.distributed.run --nproc_per_node=2 --master_port=${MASTER_PORT} ./run_clm.py --model_type forgetting_transformer --tokenizer_name gpt2 --model_name_or_path "${CKPT}" --cfg_path /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_pat310_fox_medium_docorder/cfgs/docorder.cfg --dataset_name hotpot_qa --dataset_config_name distractor --hotpot_long_jsonl /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/hotpot_long_dev_uniform_8192only.jsonl --hotpot_long_lengths 8192 --do_eval --block_size 8192 --per_device_eval_batch_size 1 --output_dir /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/results_uniform/pat310_fox_medium_docordertrain_s42_ckpt15000/L8192 --overwrite_output_dir --logging_dir /cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/results_uniform/pat310_fox_medium_docordertrain_s42_ckpt15000/L8192/log --seed 42 --load_best_model_at_end False

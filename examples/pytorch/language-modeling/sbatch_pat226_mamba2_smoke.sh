#!/bin/bash
#SBATCH --job-name=pat226_smoke
#SBATCH --output=/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/log_file/train/%j_pat226_mamba2_smoke.txt
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:a6000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00

# PAT-226 L1 smoke: Mamba-2 130M through run_clm on wikitext-103, 30 steps.
# Verifies: fla mamba2 HF registration, config overrides, finite loss,
# param count (~130M target), checkpoint save/load round trip.
# Branch: hongyusaatitech/pat-226-ssm-external-baseline-mamba-2-130m-under-the-identical-data

set -euxo pipefail

if [ -f /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh ]; then
  set +u; source /home/is/hongyu-s/miniconda3/etc/profile.d/conda.sh; conda activate latest_transformers; set -u
fi

WORKDIR=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling
cd "${WORKDIR}"

export PYTHONPATH=/project/nlp-work5/hongyu-s/transformers/src:/project/nlp-work5/hongyu-s/flash-linear-attention:${PYTHONPATH:-}
export HF_HOME=/cl/work5/hongyu-s/huggingfac
export HF_DATASETS_CACHE=/cl/work5/hongyu-s/huggingfac/datasets
export WANDB_DISABLED=true
export WANDB_MODE=disabled

RUN_OUT="${WORKDIR}/runs/pat226_mamba2/smoke"
mkdir -p "${RUN_OUT}"

python ./run_clm.py \
  --model_type mamba2 --tokenizer_name gpt2 \
  --config_overrides "hidden_size=768,num_hidden_layers=24,state_size=128,expand=2,head_dim=64,num_heads=24,vocab_size=50257,tie_word_embeddings=True,bos_token_id=50256,eos_token_id=50256,pad_token_id=50256" \
  --learning_rate 1e-4 --weight_decay 0.0 \
  --per_device_train_batch_size 4 --per_device_eval_batch_size 4 \
  --block_size 512 --dataset_name wikitext --dataset_config_name wikitext-103-raw-v1 \
  --do_train --max_steps 30 --save_steps 30 --save_safetensors False \
  --logging_steps 10 \
  --output_dir "${RUN_OUT}" --overwrite_output_dir \
  --seed 42

python - <<'EOF'
import torch, json
from transformers import AutoConfig, AutoModelForCausalLM
import fla.models  # noqa
ckpt = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat226_mamba2/smoke/checkpoint-30"
model = AutoModelForCausalLM.from_pretrained(ckpt).cuda()
n = sum(p.numel() for p in model.parameters())
print(f"[PAT-226] mamba2 param count: {n} ({n/1e6:.1f}M)")
assert 120e6 < n < 140e6, "param count outside 130M class"
ids = torch.randint(100, 5000, (1, 256)).cuda()
out = model(ids, labels=ids)
assert torch.isfinite(out.loss), "non-finite loss on reload"
print(f"[PAT-226] reload forward loss: {float(out.loss):.4f}")
print("[PAT-226] SMOKE PASS")
EOF

echo "=== PAT-226 mamba2 smoke done ==="

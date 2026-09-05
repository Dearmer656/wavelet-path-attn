#!/bin/bash
#SBATCH --job-name=fox_med_smoke
#SBATCH --output=/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/log_file/train/%j_fox_medium_smoke.txt
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:a6000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00

# PAT-226-style L1 smoke: FoX (Forgetting Transformer, Lin et al.) medium-scale config
# through run_clm on wikitext-103, 30 steps. The original PaTH-attention paper compared
# against FoX as an external baseline; adding it here at GPT-2-medium scale to match the
# QWAB-vs-baselines "identical data" comparison this session has been building
# (PA-only/Rotary/ALiBi/Mamba-2, all medium, all OWT). FoX uses fla's own
# forgetting_transformer model (fla/models/forgetting_transformer) via AutoConfig/
# AutoModelForCausalLM registration -- same mechanism already verified working for
# --model_type mamba2. Unlike Mamba-2's mamba_ssm/causal_conv1d (real CUDA kernels needing
# a dedicated mamba2_env), FoX's forgetting_attn is pure Triton, so latest_transformers
# should work without a special env -- this smoke test is exactly to confirm that.
# Verifies: fla forgetting_transformer HF registration, config overrides, finite loss,
# param count in the GPT-2-medium (~350M) class, checkpoint save/load round trip.
# hidden_size=1024/num_hidden_layers=24/num_heads=16 mirrors GPT-2-medium's shape
# (FoX's own default is 2048/24/32, sized for much larger models).
# NOTE: first attempt included pad_token_id=50256 in --config_overrides and crashed with
# "TypeError: You can only update int, float, bool or string values in the config, got
# 50256 for key pad_token_id" -- ForgettingTransformerConfig defaults pad_token_id=None,
# and HF's update_from_string type-checks against the CURRENT (None) value, not the
# annotated Optional[int] type, so it refuses the update. Dropped pad_token_id from the
# override string; not needed for this packed-sequence pretrain (no padding used).

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

RUN_OUT="${WORKDIR}/runs/pat226_fox/smoke"
mkdir -p "${RUN_OUT}"

python ./run_clm.py \
  --model_type forgetting_transformer --tokenizer_name gpt2 \
  --config_overrides "hidden_size=1024,num_hidden_layers=24,num_heads=16,vocab_size=50257,tie_word_embeddings=True,bos_token_id=50256,eos_token_id=50256" \
  --learning_rate 1e-4 --weight_decay 0.01 \
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
ckpt = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat226_fox/smoke/checkpoint-30"
model = AutoModelForCausalLM.from_pretrained(ckpt).cuda()
n = sum(p.numel() for p in model.parameters())
print(f"[PAT-226-FoX] param count: {n} ({n/1e6:.1f}M)")
assert 300e6 < n < 420e6, "param count outside GPT-2-medium class"
ids = torch.randint(100, 5000, (1, 256)).cuda()
out = model(ids, labels=ids)
assert torch.isfinite(out.loss), "non-finite loss on reload"
print(f"[PAT-226-FoX] reload forward loss: {float(out.loss):.4f}")
print("[PAT-226-FoX] SMOKE PASS")
EOF

echo "=== PAT-226 FoX medium smoke done ==="

"""Standalone wrapper: head-parallel ALiBi eager attention for HotpotQA-Long eval.

Does NOT modify any shared file (modeling_gpt2.py / run_clm.py). Monkey-patches, at the
Python level within this process only:
  1. transformers.models.gpt2.modeling_gpt2.eager_attention_forward -- replaced with a
     version that, when pe_method='alibi' and world_size>1, splits heads across ranks,
     computes each rank's local attention (O(T^2) memory divided by world_size), and
     torch.distributed.all_gather's the results back into the full [B,H,T,D] tensor
     before returning -- transparent to every caller (GPT2Attention.forward is unchanged).
     Falls through to the original function whenever world_size==1 or pe_method!='alibi',
     so this is a strict no-op outside this specific use case.
  2. torch.utils.data.distributed.DistributedSampler -- forced to act as num_replicas=1
     (no sharding), so every rank sees the identical batch, required for head-parallel
     cooperation (this ONLY matters for --do_eval-only invocations like this one; a
     --do_train run through this wrapper would also lose its train-time sharding, so this
     wrapper is not meant to be reused for training).

Env vars (set by the launching sbatch script):
  ALIBI_HEAD_PARALLEL_L: block_size / hotpot_long_lengths value (e.g. "16384")
  ALIBI_HEAD_PARALLEL_MAX_SAMPLES: --max_eval_samples value (e.g. "2000"), optional
  ALIBI_HEAD_PARALLEL_CKPT: checkpoint dir to load
"""
import os
import sys

sys.path.insert(0, "/project/nlp-work5/hongyu-s/transformers/src")
sys.path.insert(0, "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling")

import torch
import torch.distributed as dist

import transformers.models.gpt2.modeling_gpt2 as gpt2_mod
from transformers.models.gpt2.modeling_gpt2 import _get_alibi_slopes

_ORIG_EAGER = gpt2_mod.eager_attention_forward


def head_parallel_eager_attention_forward(module, query, key, value, attention_mask, head_mask=None, **kwargs):
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return _ORIG_EAGER(module, query, key, value, attention_mask, head_mask=head_mask, **kwargs)
    if getattr(getattr(module, "config", None), "pe_method", None) != "alibi":
        return _ORIG_EAGER(module, query, key, value, attention_mask, head_mask=head_mask, **kwargs)

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    H = query.size(1)
    if H % world_size != 0:
        # Can't evenly split -- fall back rather than silently produce a wrong result.
        return _ORIG_EAGER(module, query, key, value, attention_mask, head_mask=head_mask, **kwargs)
    local_H = H // world_size
    start = rank * local_H
    end = start + local_H

    q_local = query[:, start:end]
    k_local = key[:, start:end]
    v_local = value[:, start:end]

    attn_weights = torch.matmul(q_local, k_local.transpose(-1, -2))
    if module.scale_attn_weights:
        attn_weights = attn_weights / torch.full(
            [], v_local.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
        )

    q_len = query.size(-2)
    k_len = key.size(-2)
    slopes_full = _get_alibi_slopes(H, query.device).to(dtype=attn_weights.dtype)
    slopes_local = slopes_full[start:end]
    q_pos = torch.arange(k_len - q_len, k_len, device=query.device, dtype=torch.float32).unsqueeze(1)
    k_pos = torch.arange(k_len, device=query.device, dtype=torch.float32).unsqueeze(0)
    dist_mat = (q_pos - k_pos).clamp(min=0).to(dtype=attn_weights.dtype)
    alibi_bias = -slopes_local.view(local_H, 1, 1) * dist_mat.unsqueeze(0)
    attn_weights = attn_weights + alibi_bias

    if not module.is_cross_attention:
        query_length, key_length = q_len, k_len
        if module.bias.size(-1) >= key_length:
            causal_mask = module.bias[:, :, key_length - query_length : key_length, :key_length]
        else:
            causal_mask = torch.tril(
                torch.ones((query_length, key_length), dtype=torch.bool, device=attn_weights.device),
                diagonal=key_length - query_length,
            ).view(1, 1, query_length, key_length)
        mask_value = torch.finfo(attn_weights.dtype).min
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + causal_mask

    _attn_min = torch.finfo(attn_weights.dtype).min
    attn_weights = attn_weights.nan_to_num(nan=0.0, posinf=0.0, neginf=_attn_min)
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = attn_weights.type(v_local.dtype)
    attn_weights = module.attn_dropout(attn_weights)
    if head_mask is not None:
        attn_weights = attn_weights * head_mask[:, start:end]

    attn_output_local = torch.matmul(attn_weights, v_local).contiguous()  # [B, local_H, T, D]

    gathered = [torch.empty_like(attn_output_local) for _ in range(world_size)]
    dist.all_gather(gathered, attn_output_local)
    attn_output_full = torch.cat(gathered, dim=1)  # [B, H, T, D]

    attn_output_full = attn_output_full.transpose(1, 2)
    return attn_output_full, None


gpt2_mod.eager_attention_forward = head_parallel_eager_attention_forward

# Force every DistributedSampler in this process to act as num_replicas=1 (no sharding),
# so all ranks see the identical eval batch -- required for head-parallel cooperation.
import torch.utils.data.distributed as _dist_data

_OrigDistributedSampler = _dist_data.DistributedSampler


class _NoShardDistributedSampler(_OrigDistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, **kwargs):
        super().__init__(dataset, num_replicas=1, rank=0, **kwargs)


_dist_data.DistributedSampler = _NoShardDistributedSampler
import torch.utils.data as _tud

_tud.DistributedSampler = _NoShardDistributedSampler

# ---- build sys.argv for run_clm.main() ----
BSIZE = os.environ["ALIBI_HEAD_PARALLEL_L"]
CKPT = os.environ.get(
    "ALIBI_HEAD_PARALLEL_CKPT",
    "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/mix_medium_owt_alibi_10ep_s42_fp32/checkpoint-15000",
)
MAX_SAMPLES = os.environ.get("ALIBI_HEAD_PARALLEL_MAX_SAMPLES", "")
BASE = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling"
if int(BSIZE) <= 4096:
    JSONL = f"{BASE}/hotpot_long/data/hotpot_long_dev.jsonl"
else:
    JSONL = f"{BASE}/hotpot_long/data/hotpot_long_dev_uniform_{BSIZE}only.jsonl"
OUTPUT = os.environ.get("ALIBI_HEAD_PARALLEL_OUTPUT", f"{BASE}/hotpot_long/results/_head_parallel_test/L{BSIZE}")

argv = [
    sys.argv[0],
    "--model_type", "gpt2", "--tokenizer_name", "gpt2",
    "--model_name_or_path", CKPT,
    "--attn_implementation", "eager",
    "--pe_method", "alibi",
    "--bf16", "True",
    "--dataset_name", "hotpot_qa", "--dataset_config_name", "distractor",
    "--hotpot_long_jsonl", JSONL,
    "--hotpot_long_lengths", BSIZE,
    "--do_eval",
    "--block_size", BSIZE,
    "--per_device_eval_batch_size", "1",
    "--output_dir", OUTPUT, "--overwrite_output_dir",
    "--logging_dir", f"{OUTPUT}/log",
    "--ddp_timeout", "21600", "--seed", "42", "--load_best_model_at_end", "False",
]
if MAX_SAMPLES:
    argv += ["--max_eval_samples", MAX_SAMPLES]
sys.argv = argv

import run_clm

run_clm.main()

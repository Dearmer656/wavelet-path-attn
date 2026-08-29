#!/usr/bin/env python3
"""Head-parallel router-usage dump for lengths that OOM a single GPU in dense
pytorch path_attn (e.g. medium@L16384). Splits the H=16 head dim across the
launched GPUs, mirroring analysis/head_parallel_eval.py's proven L16384 eval
setup (see _gen_medium_ricker_lenbackfill/test_mix_medium_K3_ricker_128_256_384_s43_L16384_headparallel_8gpu.sh,
which validated 8xA100 for this exact checkpoint family).

Router usage capture needs NO extra gather step: _last_router_pi is computed
from hidden_states inside _build_ctxscale_shift_logit_bias_v0, and
hidden_states is never head-sliced by the parallel wrapper (only q/k/v/w/beta
are), so every rank computes the identical full-K pi independently. Only
rank 0 writes the CSV.

Launch with: torchrun --nproc_per_node=<N> dump_router_usage_headparallel.py ...
"""

import argparse
import csv
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

import sys

sys.path.insert(0, "/project/nlp-work5/hongyu-s/flash-linear-attention")
sys.path.insert(0, "/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long")

from fla.layers.path_attn import PaTHAttention

from dump_router_usage import (
    get_path_layers,
    load_cases_hotpot,
    load_path_attn_model,
    render_prompt_hotpot,
)

_orig_fn = PaTHAttention.path_attention_with_wavelet_QH


def _head_parallel_fn(self, q, k, v, w, beta, wavelet_dtt, **kwargs):
    world_size = dist.get_world_size()
    if world_size == 1:
        return _orig_fn(self, q, k, v, w, beta, wavelet_dtt, **kwargs)
    rank = dist.get_rank()
    H = w.shape[2]  # q,k,v,w,beta: (B, T, H, D)
    assert H % world_size == 0, f"num_heads={H} not divisible by world_size={world_size}"
    H_local = H // world_size
    h0, h1 = rank * H_local, (rank + 1) * H_local

    q_s = q[:, :, h0:h1]
    k_s = k[:, :, h0:h1]
    v_s = v[:, :, h0:h1]
    w_s = w[:, :, h0:h1]
    beta_s = beta[:, :, h0:h1]

    router1 = kwargs.pop("router1", None)
    router2 = kwargs.pop("router2", None)
    router1_s = router1[:, :, h0:h1] if router1 is not None else None
    router2_s = router2[:, :, h0:h1] if router2 is not None else None

    out_local = _orig_fn(
        self, q_s, k_s, v_s, w_s, beta_s, wavelet_dtt,
        router1=router1_s, router2=router2_s, **kwargs,
    )  # (B, T, H_local, D) -- as a side effect, sets self._last_router_pi

    out_local = out_local.permute(2, 0, 1, 3).contiguous()
    out_full_leading = torch.empty(
        (H,) + out_local.shape[1:], dtype=out_local.dtype, device=out_local.device
    )
    dist.all_gather_into_tensor(out_full_leading, out_local)
    return out_full_leading.permute(1, 2, 0, 3).contiguous()


PaTHAttention.path_attention_with_wavelet_QH = _head_parallel_fn


def run(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=False)
    model = load_path_attn_model(args.checkpoint, device, dtype=torch.float32)
    path_layers = get_path_layers(model)
    n_layers = len(path_layers)
    if n_layers == 0:
        raise RuntimeError("No PaTHAttention layers found.")
    for _, module in path_layers:
        module._capture_debug_tensors = False
    if rank == 0:
        print(f"Found {n_layers} PaTHAttention layers, world_size={world_size}", flush=True)

    qbin_size = args.qbin_size
    seq_lens = [int(x) for x in args.seq_lens.split(",") if x.strip()]
    all_rows = []
    K = None

    for seq_len in seq_lens:
        cases = load_cases_hotpot(args.jsonl, seq_len, args.n_case)
        if rank == 0:
            print(f"Loaded {len(cases)} cases for seq_len={seq_len}", flush=True)

        sum_usage = {}
        count = {}

        for ci, rec in enumerate(cases):
            prompt_ids = tokenizer(render_prompt_hotpot(rec), add_special_tokens=False)["input_ids"]
            answer_ids = tokenizer(f" {rec['answer']}", add_special_tokens=False)["input_ids"]
            full_ids = (prompt_ids + answer_ids)[:seq_len]
            input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                _ = model(input_ids)
            T = input_ids.shape[1]
            qbin_of_pos = torch.arange(T) // qbin_size
            for layer_idx, (_, module) in enumerate(path_layers):
                pi = getattr(module, "_last_router_pi", None)
                if pi is None:
                    raise RuntimeError(
                        f"layer {layer_idx}: missing _last_router_pi -- this checkpoint's "
                        f"wavelet_mode may not be logit_bias_ctxscale_shift_v0 (K>1 router only)."
                    )
                pi_cpu = pi[0].cpu()  # [T, K+1]
                for qb in torch.unique(qbin_of_pos).tolist():
                    sel = qbin_of_pos == qb
                    key = (layer_idx, qb)
                    vals = pi_cpu[sel].sum(dim=0)
                    if key not in sum_usage:
                        sum_usage[key] = torch.zeros_like(vals)
                        count[key] = 0
                    sum_usage[key] += vals
                    count[key] += int(sel.sum().item())
            torch.cuda.empty_cache()
            if rank == 0:
                print(f"  case {ci + 1}/{len(cases)} @ L={seq_len}", flush=True)

        K_plus_1 = next(iter(sum_usage.values())).shape[0]
        K = K_plus_1 - 1
        if rank == 0:
            for (layer_idx, qb), s in sum_usage.items():
                mean_usage = (s / max(count[(layer_idx, qb)], 1)).tolist()
                all_rows.append({
                    "model_tag": args.model_tag,
                    "seed": args.seed,
                    "seq_len": seq_len,
                    "layer": layer_idx,
                    "n_layers": n_layers,
                    "K": K,
                    "qbin": qb,
                    "qbin_lo": qb * qbin_size,
                    "qbin_hi": qb * qbin_size + qbin_size,
                    "null_usage": mean_usage[0],
                    **{f"scale{ki}_usage": mean_usage[ki + 1] for ki in range(K)},
                })

    if rank == 0:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["model_tag", "seed", "seq_len", "layer", "n_layers", "K", "qbin", "qbin_lo", "qbin_hi", "null_usage"] + [
            f"scale{ki}_usage" for ki in range(K)
        ]
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_rows)
        print(f"wrote {out_path} ({len(all_rows)} rows)")

    dist.barrier()
    dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model_tag", required=True)
    p.add_argument("--seed", required=True)
    p.add_argument("--seq_lens", default="16384")
    p.add_argument("--n_case", type=int, default=50)
    p.add_argument(
        "--jsonl",
        default="data/hotpot_long_dev_uniform_16384_large_pool.jsonl",
    )
    p.add_argument("--out_csv", required=True)
    p.add_argument("--qbin_size", type=int, default=512)
    run(p.parse_args())


if __name__ == "__main__":
    main()

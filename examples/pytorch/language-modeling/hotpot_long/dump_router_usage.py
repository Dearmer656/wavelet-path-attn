#!/usr/bin/env python3
"""
dump_router_usage.py — per-layer, per-scale QWAB router usage (K>1) for a
single checkpoint/seed. Reads module._last_router_pi (pi[...,0]=null gate,
pi[...,1:]=the K per-scale weights actually used to build the wavelet bias),
mean-pooled over sequence positions and cases, per layer.

Usage: run once per (model_tag, seed) checkpoint; combine the CSVs
downstream (plot_router_usage.py) for the small-vs-medium 3-seed comparison.
"""

import argparse
import csv
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def render_doc(title, sentences):
    return f"Title: {title}\n{' '.join(sentences).strip()}\n\n"


def render_prompt(ex):
    ctx = "".join(render_doc(t, s) for t, s in ex["context"])
    return f"Question: {ex['question']}\n\nContext:\n{ctx}Answer:"


def load_cases(jsonl_path, seq_len, n_case):
    cases = []
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec["meta"].get("target_total_tokens") == seq_len:
                cases.append(rec)
                if len(cases) >= n_case:
                    break
    return cases


def get_path_layers(model):
    path_layers = []
    for name, module in model.named_modules():
        if type(module).__name__ == "PaTHAttention":
            path_layers.append((name, module))
    return path_layers


def load_path_attn_model(checkpoint, device, dtype=torch.float32):
    config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    config.attn_implementation = "path_attn"
    with open(Path(checkpoint) / "config.json", encoding="utf-8") as f:
        raw_config = json.load(f)
    if "wavelet_ctxscale_scale_max_exp" not in raw_config:
        resolved_k = int(raw_config.get("wavelet_ctxscale_k", 8))
        if resolved_k > 1:
            config.wavelet_ctxscale_scale_max_exp = [14.0] * resolved_k
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, config=config, dtype=dtype, trust_remote_code=True,
    )
    return model.eval().to(device)


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.checkpoint} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=False)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model = load_path_attn_model(args.checkpoint, device, dtype=dtype)
    path_layers = get_path_layers(model)
    n_layers = len(path_layers)
    if n_layers == 0:
        raise RuntimeError("No PaTHAttention layers found.")
    print(f"Found {n_layers} PaTHAttention layers")
    # We only need _last_router_pi (tiny, [B,T,K+1]) -- skip the expensive
    # unconditional logits/value debug captures (full [B,H,T,T] fp32 per
    # layer, retained for all layers simultaneously) that OOM a 48GB card on
    # the medium model at L=4096.
    for _, module in path_layers:
        module._capture_debug_tensors = False

    seq_lens = [int(x) for x in args.seq_lens.split(",") if x.strip()]
    all_rows = []
    K = None

    for seq_len in seq_lens:
        cases = load_cases(args.jsonl, seq_len, args.n_case)
        print(f"Loaded {len(cases)} cases for seq_len={seq_len}")

        # sum_usage[layer] = running sum over all (case, position); count[layer] = n positions summed
        sum_usage = {li: None for li in range(n_layers)}
        count = {li: 0 for li in range(n_layers)}

        for ci, rec in enumerate(cases):
            prompt_ids = tokenizer(render_prompt(rec), add_special_tokens=False)["input_ids"]
            answer_ids = tokenizer(f" {rec['answer']}", add_special_tokens=False)["input_ids"]
            full_ids = (prompt_ids + answer_ids)[:seq_len]
            input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                _ = model(input_ids)
            for layer_idx, (_, module) in enumerate(path_layers):
                pi = getattr(module, "_last_router_pi", None)
                if pi is None:
                    raise RuntimeError(
                        f"layer {layer_idx}: missing _last_router_pi -- this checkpoint's "
                        f"wavelet_mode may not be logit_bias_ctxscale_shift_v0 (K>1 router only)."
                    )
                # pi: [B, T, K+1] -> mean over batch(=1) and T -> [K+1]
                layer_mean = pi[0].mean(dim=0).cpu()
                if sum_usage[layer_idx] is None:
                    sum_usage[layer_idx] = torch.zeros_like(layer_mean)
                sum_usage[layer_idx] += layer_mean * pi.shape[1]
                count[layer_idx] += pi.shape[1]
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"  case {ci+1}/{len(cases)} @ L={seq_len}")

        K_plus_1 = sum_usage[0].shape[0]
        K = K_plus_1 - 1
        for layer_idx in range(n_layers):
            mean_usage = (sum_usage[layer_idx] / max(count[layer_idx], 1)).tolist()
            all_rows.append({
                "model_tag": args.model_tag,
                "seed": args.seed,
                "seq_len": seq_len,
                "layer": layer_idx,
                "n_layers": n_layers,
                "K": K,
                "null_usage": mean_usage[0],
                **{f"scale{ki}_usage": mean_usage[ki + 1] for ki in range(K)},
            })

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model_tag", "seed", "seq_len", "layer", "n_layers", "K", "null_usage"] + [
        f"scale{ki}_usage" for ki in range(K)
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"wrote {out_path} ({len(all_rows)} rows)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model_tag", required=True, help="e.g. small or medium")
    p.add_argument("--seed", required=True)
    p.add_argument("--seq_lens", default="512,4096")
    p.add_argument("--n_case", type=int, default=50)
    p.add_argument("--jsonl", default="data/hotpot_long_dev.jsonl")
    p.add_argument("--out_csv", required=True)
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32",
                    help="bf16 roughly halves memory; use for the medium model at L=4096, "
                         "which OOMs a 48GB card in fp32 purely from the model's own forward "
                         "computation (independent of any debug captures).")
    run(p.parse_args())


if __name__ == "__main__":
    main()

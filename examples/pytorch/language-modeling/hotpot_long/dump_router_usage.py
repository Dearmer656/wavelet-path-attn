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


def render_prompt_hotpot(ex):
    ctx = "".join(render_doc(t, s) for t, s in ex["context"])
    return f"Question: {ex['question']}\n\nContext:\n{ctx}Answer:"


def load_cases_hotpot(jsonl_path, seq_len, n_case):
    cases = []
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec["meta"].get("target_total_tokens") == seq_len:
                cases.append(rec)
                if len(cases) >= n_case:
                    break
    return cases


# XSum: run_clm.py's prompt template (PROMPT_TPL at ~line 6138), approximated
# here -- not a byte-exact reproduction of its bucket/cache pipeline, just
# close enough for a router-usage forward pass. Uses summary_filtered (the
# factuality-filtered target) as the "answer" continuation, matching the
# hotpot prompt+answer pattern.
XSUM_PROMPT_TPL = "Summarize the following document:\n{doc}\n\nSummary:"


def render_prompt_xsum(ex):
    return XSUM_PROMPT_TPL.format(doc=ex["document"])


def load_cases_xsum(jsonl_path, n_case):
    cases = []
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("is_empty_after_filter", False):
                continue
            if not rec.get("summary_filtered"):
                continue
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
        if args.task == "xsum":
            cases = load_cases_xsum(args.jsonl, args.n_case)
        else:
            cases = load_cases_hotpot(args.jsonl, seq_len, args.n_case)
        print(f"Loaded {len(cases)} cases for seq_len={seq_len}")

        # sum_usage[(layer, qbin)] = running sum over all (case, position) in
        # that 512-token position window; count[(layer, qbin)] = n positions
        # summed. qbin = position // qbin_size, so qbin width is fixed at
        # qbin_size tokens regardless of seq_len (finer than a fixed 3-way
        # early/mid/late split).
        qbin_size = args.qbin_size
        sum_usage = {}
        count = {}

        for ci, rec in enumerate(cases):
            if args.task == "xsum":
                prompt_ids = tokenizer(render_prompt_xsum(rec), add_special_tokens=False)["input_ids"]
                answer_ids = tokenizer(f" {rec['summary_filtered']}", add_special_tokens=False)["input_ids"]
            else:
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
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"  case {ci+1}/{len(cases)} @ L={seq_len}")

        K_plus_1 = next(iter(sum_usage.values())).shape[0]
        K = K_plus_1 - 1
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model_tag", required=True, help="e.g. small or medium")
    p.add_argument("--seed", required=True)
    p.add_argument("--task", choices=["hotpot", "xsum"], default="hotpot")
    p.add_argument("--seq_lens", default="512,4096")
    p.add_argument("--n_case", type=int, default=50)
    p.add_argument(
        "--jsonl",
        default="data/hotpot_long_dev.jsonl",
        help="hotpot: data/hotpot_long_dev.jsonl (default). xsum: pass "
             "/cl/work5/hongyu-s/fact-check-summarization/xsum_test_filter_level2_official_style.jsonl",
    )
    p.add_argument("--out_csv", required=True)
    p.add_argument("--qbin_size", type=int, default=512,
                    help="position-bin width in tokens; usage is aggregated per "
                         "[qbin*qbin_size, (qbin+1)*qbin_size) window instead of "
                         "collapsed over the whole sequence.")
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32",
                    help="bf16 roughly halves memory; use for the medium model at L=4096, "
                         "which OOMs a 48GB card in fp32 purely from the model's own forward "
                         "computation (independent of any debug captures).")
    run(p.parse_args())


if __name__ == "__main__":
    main()

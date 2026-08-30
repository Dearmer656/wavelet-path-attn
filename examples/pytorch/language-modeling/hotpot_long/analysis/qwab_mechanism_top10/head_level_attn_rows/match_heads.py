#!/usr/bin/env python3
"""Match PA-only heads to QWAB heads by function before comparing attention
patterns -- head index is not a stable identity across two independently
trained models (different init/optimization trajectory can permute which
index ends up playing which functional role).

Signature (per head, avoids circularity with the divergence metric we
actually care about): a log-spaced relative-distance attention-mass
histogram, averaged over many query rows across several DIFFERENT example
cases (not the same case used later for divergence inspection). This
captures a head's "personality" (local/vertical vs long-range/distributed)
independent of any specific PA-vs-QWAB row-level disagreement.

Matching: cosine similarity between every PA head and every QWAB head's
signature, then optimal one-to-one assignment via the Hungarian algorithm
(scipy.optimize.linear_sum_assignment) maximizing total similarity.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

import sys

sys.path.insert(0, "/project/nlp-work5/hongyu-s/flash-linear-attention")
sys.path.insert(0, "/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long")

from fla.layers.path_attn import PaTHAttention  # noqa: F401
from dump_router_usage import render_prompt_hotpot, load_cases_hotpot, get_path_layers, load_path_attn_model
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# log-spaced distance-from-query bins (in tokens)
DIST_BINS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


def causal_softmax(logits_hTT: torch.Tensor) -> torch.Tensor:
    H, T, _ = logits_hTT.shape
    i_idx = torch.arange(T, device=logits_hTT.device).view(T, 1)
    j_idx = torch.arange(T, device=logits_hTT.device).view(1, T)
    causal = (j_idx <= i_idx)
    masked = logits_hTT.masked_fill(~causal.unsqueeze(0), float("-inf"))
    return torch.softmax(masked, dim=-1)


def distance_histogram(attn_row: torch.Tensor, q: int) -> np.ndarray:
    """attn_row: [q+1] attention weights over keys 0..q. Returns mass per
    log-distance bin (distance = q - key_pos)."""
    dist = q - torch.arange(q + 1, device=attn_row.device)
    hist = np.zeros(len(DIST_BINS) - 1)
    dist_np = dist.cpu().numpy()
    attn_np = attn_row.cpu().numpy()
    for bi in range(len(DIST_BINS) - 1):
        lo, hi = DIST_BINS[bi], DIST_BINS[bi + 1]
        mask = (dist_np >= lo) & (dist_np < hi)
        hist[bi] = attn_np[mask].sum()
    return hist


def build_signature(model, path_layers, tokenizer, cases, layer_idx, seq_len, device, query_rows):
    """Returns [H, n_bins] signature matrix, averaged over cases and query rows."""
    H = None
    acc = None
    n = 0
    for rec in cases:
        prompt_ids = tokenizer(render_prompt_hotpot(rec), add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(f" {rec['answer']}", add_special_tokens=False)["input_ids"]
        full_ids = (prompt_ids + answer_ids)[:seq_len]
        if len(full_ids) < seq_len:
            continue
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            _ = model(input_ids)
        logits = path_layers[layer_idx][1]._last_logits_pa_only[0]  # PA-only always uses this; QWAB caller passes _last_logits_full
        attn = causal_softmax(logits)  # [H,T,T]
        if H is None:
            H = attn.shape[0]
            acc = np.zeros((H, len(DIST_BINS) - 1))
        for h in range(H):
            for q in query_rows:
                if q >= attn.shape[1]:
                    continue
                acc[h] += distance_histogram(attn[h, q, : q + 1], q)
                n += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    acc /= max(n / H, 1)
    return acc  # [H, n_bins], row-normalized-ish (not exactly, but comparable)


def build_signature_qwab(model, path_layers, tokenizer, cases, layer_idx, seq_len, device, query_rows):
    H = None
    acc = None
    n = 0
    for rec in cases:
        prompt_ids = tokenizer(render_prompt_hotpot(rec), add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(f" {rec['answer']}", add_special_tokens=False)["input_ids"]
        full_ids = (prompt_ids + answer_ids)[:seq_len]
        if len(full_ids) < seq_len:
            continue
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            _ = model(input_ids)
        logits = path_layers[layer_idx][1]._last_logits_full[0]  # Q-on: real bias active
        attn = causal_softmax(logits)
        if H is None:
            H = attn.shape[0]
            acc = np.zeros((H, len(DIST_BINS) - 1))
        for h in range(H):
            for q in query_rows:
                if q >= attn.shape[1]:
                    continue
                acc[h] += distance_histogram(attn[h, q, : q + 1], q)
                n += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    acc /= max(n / H, 1)
    return acc


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.pa_checkpoint, use_fast=False)

    print(f"Loading PA-only: {args.pa_checkpoint}")
    pa_model = load_path_attn_model(args.pa_checkpoint, device, dtype=torch.float32)
    pa_layers = get_path_layers(pa_model)
    for _, m in pa_layers:
        m._capture_debug_tensors = True

    print(f"Loading QWAB: {args.qwab_checkpoint}")
    qwab_model = load_path_attn_model(args.qwab_checkpoint, device, dtype=torch.float32)
    qwab_layers = get_path_layers(qwab_model)
    for _, m in qwab_layers:
        m._capture_debug_tensors = True

    all_cases = load_cases_hotpot(args.jsonl, args.seq_len, args.n_case_fit + args.n_case_holdout)
    fit_cases = all_cases[: args.n_case_fit]
    holdout_cases = all_cases[args.n_case_fit: args.n_case_fit + args.n_case_holdout]
    print(f"fit_cases={len(fit_cases)} holdout_cases={len(holdout_cases)}")

    query_rows = list(range(511, args.seq_len, 512))

    print("Building PA signature...")
    pa_sig = build_signature(pa_model, pa_layers, tokenizer, fit_cases, args.layer, args.seq_len, device, query_rows)
    print("Building QWAB signature...")
    qwab_sig = build_signature_qwab(qwab_model, qwab_layers, tokenizer, fit_cases, args.layer, args.seq_len, device, query_rows)

    # cosine similarity matrix [H_pa, H_qwab]
    def norm(x):
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)

    pa_n = norm(pa_sig)
    qwab_n = norm(qwab_sig)
    sim = pa_n @ qwab_n.T  # [H,H]

    row_ind, col_ind = linear_sum_assignment(-sim)  # maximize similarity
    matching = list(zip(row_ind.tolist(), col_ind.tolist(), [float(sim[r, c]) for r, c in zip(row_ind, col_ind)]))
    matching.sort(key=lambda x: -x[2])

    print("\nPA_head -> QWAB_head (cosine sim), sorted by match quality:")
    for pa_h, qw_h, s in matching:
        flag = " <-- WEAK MATCH" if s < args.weak_thresh else ""
        print(f"  PA head {pa_h:2d} <-> QWAB head {qw_h:2d}   sim={s:.3f}{flag}")

    out = {
        "layer": args.layer,
        "seq_len": args.seq_len,
        "n_case_fit": len(fit_cases),
        "dist_bins": DIST_BINS,
        "pa_signature": pa_sig.tolist(),
        "qwab_signature": qwab_sig.tolist(),
        "similarity_matrix": sim.tolist(),
        "matching": [{"pa_head": p, "qwab_head": q, "cosine_sim": s} for p, q, s in matching],
    }
    out_path = OUT_DIR / f"head_matching_layer{args.layer}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pa_checkpoint", required=True)
    p.add_argument("--qwab_checkpoint", required=True)
    p.add_argument("--seq_len", type=int, default=4096)
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--n_case_fit", type=int, default=5)
    p.add_argument("--n_case_holdout", type=int, default=5)
    p.add_argument("--weak_thresh", type=float, default=0.5)
    p.add_argument("--jsonl", default="/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/hotpot_long_dev.jsonl")
    run(p.parse_args())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Head-level attention-score row comparison, PA-only vs QWAB (Q-on), small
model, L4096. Motivation: aggregate (layer/head-averaged) logit-difference
statistics in PAT-254 Tasks 1-3/section-3 kept coming back weak/null despite
a clear, real F1 gap at L4096 -- averaging across 12 heads may be vanishing a
real per-head effect. This does the opposite: NO averaging, one query row at
a time, one head at a time, plotting the actual post-softmax attention score
over all causal key positions.

Query rows sampled every 512 tokens (511, 1023, 1535, ..., last valid row),
i.e. the last position of each 512-token block, so each row has a full causal
key range aligned to the block boundaries.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import sys

sys.path.insert(0, "/project/nlp-work5/hongyu-s/flash-linear-attention")
sys.path.insert(0, "/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long")

from fla.layers.path_attn import PaTHAttention  # noqa: F401
from dump_router_usage import render_prompt_hotpot, load_cases_hotpot, get_path_layers, load_path_attn_model
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def causal_softmax(logits_hTT: torch.Tensor) -> torch.Tensor:
    """logits_hTT: [H,T,T] raw logits. Returns [H,T,T] causal-masked softmax."""
    H, T, _ = logits_hTT.shape
    i_idx = torch.arange(T, device=logits_hTT.device).view(T, 1)
    j_idx = torch.arange(T, device=logits_hTT.device).view(1, T)
    causal = (j_idx <= i_idx)
    masked = logits_hTT.masked_fill(~causal.unsqueeze(0), float("-inf"))
    return torch.softmax(masked, dim=-1)


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

    cases = load_cases_hotpot(args.jsonl, args.seq_len, args.n_case)
    rec = cases[args.case_idx]
    prompt_ids = tokenizer(render_prompt_hotpot(rec), add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(f" {rec['answer']}", add_special_tokens=False)["input_ids"]
    full_ids = (prompt_ids + answer_ids)[: args.seq_len]
    if len(full_ids) < args.seq_len:
        raise RuntimeError(f"case {args.case_idx} too short: {len(full_ids)} < {args.seq_len}")
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    T = args.seq_len

    with torch.no_grad():
        _ = pa_model(input_ids)
    pa_logits = [m._last_logits_pa_only[0].clone() for _, m in pa_layers]  # per layer [H,T,T]

    with torch.no_grad():
        _ = qwab_model(input_ids)
    qwab_logits = [m._last_logits_full[0].clone() for _, m in qwab_layers]  # Q-on, real bias active

    n_layers = len(pa_logits)
    layer_idx = args.layer
    H = pa_logits[layer_idx].shape[0]
    heads = list(range(H)) if args.heads == "all" else [int(h) for h in args.heads.split(",")]

    query_rows = list(range(511, T, 512))
    print(f"layer={layer_idx} heads={heads} query_rows={query_rows}")

    pa_attn = causal_softmax(pa_logits[layer_idx])   # [H,T,T]
    qwab_attn = causal_softmax(qwab_logits[layer_idx])

    n_rows = len(heads)
    n_cols = len(query_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 2.2 * n_rows), squeeze=False)
    for hi, h in enumerate(heads):
        for qi, q in enumerate(query_rows):
            ax = axes[hi][qi]
            keys = np.arange(q + 1)
            pa_row = pa_attn[h, q, : q + 1].cpu().numpy()
            qwab_row = qwab_attn[h, q, : q + 1].cpu().numpy()
            ax.plot(keys, pa_row, color="tab:blue", lw=0.8, label="PA-only" if (hi == 0 and qi == 0) else None)
            ax.plot(keys, qwab_row, color="tab:orange", lw=0.8, alpha=0.8, label="QWAB (Q-on)" if (hi == 0 and qi == 0) else None)
            if hi == 0:
                ax.set_title(f"q={q}", fontsize=9)
            if qi == 0:
                ax.set_ylabel(f"head {h}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.legend(loc="upper right", fontsize=9)
    fig.suptitle(f"Attention score per head, per query row (every 512 tok), layer={layer_idx}, L={T}, case={args.case_idx}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = OUT_DIR / f"head_attn_rows_layer{layer_idx}_case{args.case_idx}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")

    # Also dump per-head, per-query-row summary stats (max-attn key position, entropy, PA-vs-QWAB L1 diff)
    print("\nsummary (head, query, PA_argmax_key, QWAB_argmax_key, PA_entropy, QWAB_entropy, L1_diff):")
    for h in heads:
        for q in query_rows:
            pa_row = pa_attn[h, q, : q + 1]
            qwab_row = qwab_attn[h, q, : q + 1]
            pa_am = int(pa_row.argmax().item())
            qw_am = int(qwab_row.argmax().item())
            pa_ent = float(-(pa_row * (pa_row + 1e-12).log()).sum().item())
            qw_ent = float(-(qwab_row * (qwab_row + 1e-12).log()).sum().item())
            l1 = float((pa_row - qwab_row).abs().sum().item())
            print(f"  h={h} q={q} PA_argmax={pa_am} QWAB_argmax={qw_am} PA_ent={pa_ent:.3f} QWAB_ent={qw_ent:.3f} L1={l1:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pa_checkpoint", required=True)
    p.add_argument("--qwab_checkpoint", required=True)
    p.add_argument("--seq_len", type=int, default=4096)
    p.add_argument("--case_idx", type=int, default=0)
    p.add_argument("--n_case", type=int, default=5)
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--heads", default="all")
    p.add_argument("--jsonl", default="/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/hotpot_long_dev.jsonl")
    run(p.parse_args())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
dump_pathonly_vs_qwab_attn_bins.py — compare a genuinely-independently-trained
PaTH-only checkpoint against a separately-trained PaTH+QWAB checkpoint in
query-relative-position bins x key-lag bins.

Unlike dump_qwab_attn_bins.py (which decomposes a SINGLE QWAB checkpoint into
its own base-logits vs full-logits components), this script uses TWO
checkpoints so the comparison captures both QWAB's additive bias AND any
difference in how the PaTH backbone itself was shaped by training with QWAB
present vs absent -- the within-checkpoint decomposition only sees the
former.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from dump_qwab_attn_bins import (
    aggregate_head_mean,
    bin_attention_mass,
    build_lag_bins,
    causal_mask,
    compute_probs,
    get_path_layers,
    load_cases,
    mean_kl,
    plot_heatmap,
    plot_overlay,
    render_prompt,
    write_csv,
)


def load_path_attn_model(checkpoint, device):
    config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    config.attn_implementation = "path_attn"
    # Pre-wavelet_ctxscale_k-sweep checkpoints (e.g. a PA-only baseline with
    # bias_type=None) have neither wavelet_ctxscale_k nor
    # wavelet_ctxscale_scale_max_exp in their saved config.json. fla's class
    # default for K is 8 (_PaTHAttention.__init__), and a checkpoint's saved
    # weights are shaped at that true K -- do NOT override K. But
    # wavelet_ctxscale_scale_max_exp's own default (14.0, a bare scalar) is
    # only valid for K=1, so with K=8 it fails the internal shape check;
    # supply a validly-shaped list only when the checkpoint didn't save its
    # own value (never override an explicit config).
    import json

    with open(Path(checkpoint) / "config.json", encoding="utf-8") as f:
        raw_config = json.load(f)
    if "wavelet_ctxscale_scale_max_exp" not in raw_config:
        resolved_k = int(raw_config.get("wavelet_ctxscale_k", 8))
        if resolved_k > 1:
            config.wavelet_ctxscale_scale_max_exp = [14.0] * resolved_k
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        config=config,
        dtype=torch.float32,
        trust_remote_code=True,
    )
    return model.eval().to(device)


def build_input_ids(tokenizer, rec, seq_len, device):
    prompt_ids = tokenizer(render_prompt(rec), add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(f" {rec['answer']}", add_special_tokens=False)["input_ids"]
    full_ids = (prompt_ids + answer_ids)[:seq_len]
    return torch.tensor([full_ids], dtype=torch.long, device=device), len(full_ids)


def forward_only(model, input_ids):
    # Populates every PaTHAttention layer's _last_logits_full module attribute
    # as a side effect. Do NOT eagerly materialize softmax probs for all
    # layers here (as an earlier version of this script did) -- holding two
    # models' worth of all-12-layers [H,L,L] raw buffers AND all-12-layers
    # softmax outputs simultaneously at L=4096 is ~55GB, more than a 48GB
    # card. Read+softmax one layer at a time instead (see layer_probs below),
    # so only the unavoidable raw buffers (~18.4GB/model at L=4096) plus one
    # layer's transient softmax output are ever live at once.
    with torch.no_grad():
        _ = model(input_ids)


def layer_probs(module, layer_idx, L, upper, scale):
    buf = getattr(module, "_last_logits_full", None)
    if buf is None:
        raise RuntimeError(f"layer {layer_idx}: missing _last_logits_full capture buffer")
    return compute_probs(buf[0, :, :L, :L], upper, scale)


def run(args):
    seq_lens = [int(x) for x in args.seq_lens.split(",") if x.strip()]
    n_case = int(args.n_case)
    n_qbin = int(args.n_qbin)
    top_k_layers = int(args.top_k_layers)
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading PA-only model {args.checkpoint_paonly} on {device}")
    tok_pa = AutoTokenizer.from_pretrained(args.checkpoint_paonly, use_fast=False)
    model_pa = load_path_attn_model(args.checkpoint_paonly, device)
    layers_pa = get_path_layers(model_pa)

    print(f"Loading QWAB model {args.checkpoint_qwab} on {device}")
    tok_qwab = AutoTokenizer.from_pretrained(args.checkpoint_qwab, use_fast=False)
    model_qwab = load_path_attn_model(args.checkpoint_qwab, device)
    layers_qwab = get_path_layers(model_qwab)

    n_layers = len(layers_pa)
    if n_layers == 0 or len(layers_qwab) != n_layers:
        raise RuntimeError(f"PaTHAttention layer count mismatch: pa={len(layers_pa)} qwab={len(layers_qwab)}")
    print(f"Found {n_layers} PaTHAttention layers in both models")

    scan_sum = defaultdict(lambda: np.zeros(2, dtype=np.float64))
    all_cases = {}
    layer_scores = defaultdict(list)

    for seq_len in seq_lens:
        cases = load_cases(args.jsonl, seq_len, n_case)
        all_cases[seq_len] = cases
        print(f"Loaded {len(cases)} cases for seq_len={seq_len}")
        for ci, rec in enumerate(cases):
            input_ids_pa, L_pa = build_input_ids(tok_pa, rec, seq_len, device)
            input_ids_qwab, L_qwab = build_input_ids(tok_qwab, rec, seq_len, device)
            if L_pa != L_qwab:
                raise RuntimeError(f"tokenization length mismatch: pa={L_pa} qwab={L_qwab} (different tokenizers?)")
            L = L_pa
            upper = causal_mask(L, device)
            scale_pa = (model_pa.config.n_embd // model_pa.config.n_head) ** -0.5
            scale_qwab = (model_qwab.config.n_embd // model_qwab.config.n_head) ** -0.5
            forward_only(model_pa, input_ids_pa)
            forward_only(model_qwab, input_ids_qwab)
            for layer_idx in range(n_layers):
                p_base = layer_probs(layers_pa[layer_idx][1], layer_idx, L, upper, scale_pa)
                p_wav = layer_probs(layers_qwab[layer_idx][1], layer_idx, L, upper, scale_qwab)
                kl = mean_kl(p_wav, p_base)
                key = (seq_len, layer_idx)
                scan_sum[key][0] += kl
                scan_sum[key][1] += 1.0
                layer_scores[layer_idx].append(kl)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"  case {ci+1}/{len(cases)}")

    scan_path = out_dir / "layer_kl_scan.csv"
    scan_rows = []
    for seq_len in seq_lens:
        for layer_idx in range(n_layers):
            s, c = scan_sum[(seq_len, layer_idx)]
            scan_rows.append({
                "length": seq_len,
                "layer": layer_idx,
                "mean_kl": float(s / max(c, 1.0)),
            })
    with open(scan_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["length", "layer", "mean_kl"])
        w.writeheader()
        w.writerows(scan_rows)
    print(f"wrote {scan_path}")
    print("\nLayer KL scan (KL(QWAB || PA-only), two independently-trained checkpoints):")
    print(f"{'layer':>5} {'mean_kl@all':>12} {'by_len':>24}")
    layer_avg = []
    for layer_idx in range(n_layers):
        vals = layer_scores[layer_idx]
        by_len = []
        for seq_len in seq_lens:
            m = [r["mean_kl"] for r in scan_rows if r["layer"] == layer_idx and r["length"] == seq_len]
            by_len.append(sum(m) / max(len(m), 1))
        avg = sum(vals) / max(len(vals), 1)
        layer_avg.append((avg, layer_idx))
        print(f"{layer_idx:5d} {avg:12.6f} {str([round(x, 6) for x in by_len]):>24}")

    top_layers = [idx for _, idx in sorted(layer_avg, reverse=True)[:top_k_layers]]
    selected_layers = []
    for idx in [0, n_layers - 1] + top_layers:
        if idx not in selected_layers:
            selected_layers.append(idx)
    print(f"Selected layers for detailed binning: {selected_layers}")

    bin_sum = defaultdict(lambda: np.zeros(3, dtype=np.float64))
    for seq_len in seq_lens:
        cases = all_cases[seq_len]
        for ci, rec in enumerate(cases):
            input_ids_pa, L_pa = build_input_ids(tok_pa, rec, seq_len, device)
            input_ids_qwab, L_qwab = build_input_ids(tok_qwab, rec, seq_len, device)
            if L_pa != L_qwab:
                raise RuntimeError(f"tokenization length mismatch: pa={L_pa} qwab={L_qwab}")
            L = L_pa
            upper = causal_mask(L, device)
            scale_pa = (model_pa.config.n_embd // model_pa.config.n_head) ** -0.5
            scale_qwab = (model_qwab.config.n_embd // model_qwab.config.n_head) ** -0.5
            forward_only(model_pa, input_ids_pa)
            forward_only(model_qwab, input_ids_qwab)
            for layer_idx in range(n_layers):
                if layer_idx not in selected_layers:
                    continue
                p_base = layer_probs(layers_pa[layer_idx][1], layer_idx, L, upper, scale_pa)
                p_wav = layer_probs(layers_qwab[layer_idx][1], layer_idx, L, upper, scale_qwab)
                for h in range(p_base.shape[0]):
                    rows = bin_attention_mass(p_base[h:h+1], p_wav[h:h+1], n_qbin)
                    for r in rows:
                        key = (seq_len, layer_idx, h, r[1], r[2], r[3], r[4], r[5], r[6])
                        bin_sum[key][0] += r[7]
                        bin_sum[key][1] += r[8]
                        bin_sum[key][2] += r[9]
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"  binning case {ci+1}/{len(cases)} @ L={seq_len}")

    bins_rows_all = []
    for key, (sum_base, sum_wav, n_rows) in bin_sum.items():
        seq_len, layer_idx, h, qbin, qlo, qhi, lagbin, lo, hi = key
        mean_base = sum_base / max(n_rows, 1.0)
        mean_wav = sum_wav / max(n_rows, 1.0)
        bins_rows_all.append({
            "length": seq_len,
            "layer": layer_idx,
            "head": h,
            "qbin": qbin,
            "qpos_lo": qlo,
            "qpos_hi": qhi,
            "lagbin": lagbin,
            "lag_lo": lo,
            "lag_hi": hi,
            "sum_mass_base": float(sum_base),
            "sum_mass_wav": float(sum_wav),
            "mean_mass_base": float(mean_base),
            "mean_mass_wav": float(mean_wav),
            "delta": float(mean_wav - mean_base),
            "n_rows": int(n_rows),
        })
    for seq_len in seq_lens:
        for layer_idx in selected_layers:
            rows = [r for r in bins_rows_all if r["length"] == seq_len and r["layer"] == layer_idx]
            head_mean = aggregate_head_mean(rows)
            all_rows = rows + head_mean
            csv_path = out_dir / f"bins_L{seq_len}_layer{layer_idx:02d}.csv"
            write_csv(csv_path, all_rows)
            print(f"wrote {csv_path}  ({len(all_rows)} rows)")
            plot_overlay(all_rows, fig_dir / f"layer{layer_idx:02d}_L{seq_len}_bins_overlay.png",
                         layer_idx, seq_len, n_qbin)
            plot_heatmap(all_rows, fig_dir / f"layer{layer_idx:02d}_L{seq_len}_delta_heatmap.png",
                         layer_idx, seq_len, n_qbin)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint_paonly",
        default="/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/PA_baseline_multi_seeds/token_even_mix_PA_s42/checkpoint-15000",
    )
    p.add_argument(
        "--checkpoint_qwab",
        default="/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat244_dual_temp/K1_L512_me16_rho256_ricker_s42/checkpoint-15000",
    )
    p.add_argument("--seq_lens", default="512,4096")
    p.add_argument("--n_case", type=int, default=50)
    p.add_argument("--jsonl", default="data/hotpot_long_dev.jsonl")
    p.add_argument("--n_qbin", type=int, default=10)
    p.add_argument("--out_dir", default="analysis_outputs/pathonly_vs_qwab_attn_bins")
    p.add_argument("--top_k_layers", type=int, default=3)
    run(p.parse_args())


if __name__ == "__main__":
    main()

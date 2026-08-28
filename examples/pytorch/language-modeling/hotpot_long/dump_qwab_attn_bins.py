#!/usr/bin/env python3
"""
dump_qwab_attn_bins.py — compare PaTH-only vs PaTH+QWAB attention mass in
query-relative-position bins x key-lag bins, using a single QWAB checkpoint.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def causal_mask(L, device):
    return torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)


def get_path_layers(model):
    path_layers = []
    for name, module in model.named_modules():
        if type(module).__name__ == "PaTHAttention":
            path_layers.append((name, module))
    return path_layers


def compute_probs(logits_raw, upper, scale):
    logits = logits_raw.to(torch.float32).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    # IMPORTANT: the captured logits are UNSCALED, so we must apply the
    # per-head scale here before masking/softmax or the distribution becomes
    # artificially over-peaked.
    logits = logits * scale
    neg = torch.finfo(logits.dtype).min
    logits = logits.masked_fill(upper.unsqueeze(0), neg)
    return torch.log_softmax(logits, dim=-1).exp()


def mean_kl(p_wav, p_base):
    log_wav = torch.log(p_wav.clamp_min(1e-45))
    log_base = torch.log(p_base.clamp_min(1e-45))
    kl_row = (p_wav * (log_wav - log_base)).sum(dim=-1)
    return kl_row[:, 1:].mean().item()


def make_qbin_of_row(L, n_qbin, device):
    return torch.clamp((torch.arange(L, device=device).float() / L * n_qbin).long(),
                       max=n_qbin - 1)


def build_lag_bins():
    edges = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def bin_attention_mass(P_base, P_wav, n_qbin):
    H, L, _ = P_base.shape
    qbin_of_row = make_qbin_of_row(L, n_qbin, P_base.device)
    lag_bins = build_lag_bins()
    lag = torch.arange(L, device=P_base.device)

    rows = []
    for qbin, (qlo, qhi) in enumerate([(i / n_qbin, (i + 1) / n_qbin) for i in range(n_qbin)]):
        row_sel = torch.nonzero(qbin_of_row == qbin, as_tuple=False).flatten()
        row_sel = row_sel[row_sel > 0]
        n_rows = int(row_sel.numel())
        if n_rows == 0:
            for h in range(H):
                for lagbin, (lo, hi) in enumerate(lag_bins):
                    rows.append((h, qbin, qlo, qhi, lagbin, lo, hi, 0.0, 0.0, 0))
            continue
        for lagbin, (lo, hi) in enumerate(lag_bins):
            key_sel = (lag >= lo) & (lag < hi)
            if not bool(key_sel.any()):
                mass_base = torch.zeros((H, n_rows), device=P_base.device)
                mass_wav = mass_base
            else:
                mass_base = P_base[:, row_sel][:, :, key_sel].sum(dim=-1)
                mass_wav = P_wav[:, row_sel][:, :, key_sel].sum(dim=-1)
            for h in range(H):
                rows.append((h, qbin, qlo, qhi, lagbin, lo, hi,
                             float(mass_base[h].sum().item()), float(mass_wav[h].sum().item()),
                             n_rows))
    return rows


def aggregate_head_mean(rows):
    by_key = {}
    for r in rows:
        key = (r["length"], r["layer"], r["qbin"], r["qpos_lo"], r["qpos_hi"],
               r["lagbin"], r["lag_lo"], r["lag_hi"])
        by_key.setdefault(key, []).append(r)
    out = []
    for key, items in by_key.items():
        sum_base = sum(x["sum_mass_base"] for x in items)
        sum_wav = sum(x["sum_mass_wav"] for x in items)
        n_rows = sum(x["n_rows"] for x in items)
        mean_base = sum_base / max(n_rows, 1)
        mean_wav = sum_wav / max(n_rows, 1)
        out.append({
            "length": key[0],
            "layer": key[1],
            "head": -1,
            "qbin": key[2],
            "qpos_lo": key[3],
            "qpos_hi": key[4],
            "lagbin": key[5],
            "lag_lo": key[6],
            "lag_hi": key[7],
            "sum_mass_base": float(sum_base),
            "sum_mass_wav": float(sum_wav),
            "mean_mass_base": float(mean_base),
            "mean_mass_wav": float(mean_wav),
            "delta": float(mean_wav - mean_base),
            "n_rows": n_rows,
        })
    return out


def write_csv(path, rows):
    fieldnames = ["length", "layer", "head", "qbin", "qpos_lo", "qpos_hi",
                  "lagbin", "lag_lo", "lag_hi", "mean_mass_base",
                  "mean_mass_wav", "delta", "n_rows"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def plot_overlay(rows, out_path, layer, seq_len, n_qbin):
    rows = [r for r in rows if r["head"] == -1]
    lagbins = sorted({r["lagbin"] for r in rows})
    qbins = sorted({r["qbin"] for r in rows})
    n_panels = len(qbins)
    ncols = 5 if n_panels > 1 else 1
    nrows = int(math.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows), squeeze=False)
    axes = axes.flatten()
    for ax in axes[n_panels:]:
        ax.axis("off")
    for i, qbin in enumerate(qbins):
        ax = axes[i]
        panel = [r for r in rows if r["qbin"] == qbin]
        panel = sorted(panel, key=lambda x: x["lagbin"])
        x = [r["lagbin"] for r in panel]
        yb = [r["mean_mass_base"] for r in panel]
        yw = [r["mean_mass_wav"] for r in panel]
        ax.plot(x, yb, label="PA-only", lw=2)
        ax.plot(x, yw, label="PaTH+QWAB", lw=2, ls="--")
        qlo, qhi = panel[0]["qpos_lo"], panel[0]["qpos_hi"]
        ax.set_title(f"qbin {qbin} [{qlo:.1f}, {qhi:.1f})")
        ax.set_xlabel("lag bin")
        ax.set_ylabel("mean mass")
        ax.set_xticks(lagbins)
        ax.set_xticklabels([str(b) for b in lagbins], rotation=45, ha="right")
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    fig.suptitle(f"Layer {layer:02d} L={seq_len} attention mass by lag bin")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_heatmap(rows, out_path, layer, seq_len, n_qbin):
    rows = [r for r in rows if r["head"] == -1]
    qbins = sorted({r["qbin"] for r in rows})
    lagbins = sorted({r["lagbin"] for r in rows})
    mat = np.zeros((len(qbins), len(lagbins)), dtype=np.float64)
    for r in rows:
        mat[r["qbin"], r["lagbin"]] = r["delta"]
    vmax = np.max(np.abs(mat)) if mat.size else 1.0
    fig, ax = plt.subplots(figsize=(1.1 * len(lagbins) + 2.5, 0.5 * len(qbins) + 2.5))
    im = ax.imshow(mat, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xlabel("lag bin")
    ax.set_ylabel("qbin")
    ax.set_xticks(range(len(lagbins)))
    ax.set_yticks(range(len(qbins)))
    ax.set_xticklabels([str(b) for b in lagbins], rotation=45, ha="right")
    ax.set_yticklabels([str(b) for b in qbins])
    ax.set_title(f"Layer {layer:02d} L={seq_len} delta (QWAB - PA)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run(args):
    checkpoint = str(args.checkpoint)
    seq_lens = [int(x) for x in args.seq_lens.split(",") if x.strip()]
    n_case = int(args.n_case)
    n_qbin = int(args.n_qbin)
    top_k_layers = int(args.top_k_layers)
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model {checkpoint} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
    # NOTE: passing attn_implementation="path_attn" directly to from_pretrained()
    # now fails HF's strict attn-implementation whitelist (eager/flash/sdpa only).
    # The working pattern (matches run_clm.py's model loading) is to set it as a
    # plain attribute on the config object *before* from_pretrained, and not pass
    # attn_implementation as a kwarg at all -- the custom PaTH/QWAB code reads it
    # back via getattr(config, "attn_implementation", None).
    config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    config.attn_implementation = "path_attn"
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        config=config,
        dtype=torch.float32,
        trust_remote_code=True,
    ).eval().to(device)

    path_layers = get_path_layers(model)
    n_layers = len(path_layers)
    if n_layers == 0:
        raise RuntimeError("No PaTHAttention layers found; check PYTHONPATH/model.")
    print(f"Found {n_layers} PaTHAttention layers")

    scan_sum = defaultdict(lambda: np.zeros(2, dtype=np.float64))
    all_cases = {}
    layer_scores = defaultdict(list)

    for seq_len in seq_lens:
        cases = load_cases(args.jsonl, seq_len, n_case)
        all_cases[seq_len] = cases
        print(f"Loaded {len(cases)} cases for seq_len={seq_len}")
        for ci, rec in enumerate(cases):
            prompt_ids = tokenizer(render_prompt(rec), add_special_tokens=False)["input_ids"]
            answer_ids = tokenizer(f" {rec['answer']}", add_special_tokens=False)["input_ids"]
            full_ids = (prompt_ids + answer_ids)[:seq_len]
            L = len(full_ids)
            input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                _ = model(input_ids)
            upper = causal_mask(L, device)
            scale = (model.config.n_embd // model.config.n_head) ** -0.5
            for layer_idx, (_, module) in enumerate(path_layers):
                base = getattr(module, "_last_logits_pa_only", None)
                full = getattr(module, "_last_logits_full", None)
                if base is None or full is None:
                    raise RuntimeError(f"layer {layer_idx}: missing capture buffers")
                p_base = compute_probs(base[0, :, :L, :L], upper, scale)
                p_wav = compute_probs(full[0, :, :L, :L], upper, scale)
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
    print("\nLayer KL scan:")
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
            prompt_ids = tokenizer(render_prompt(rec), add_special_tokens=False)["input_ids"]
            answer_ids = tokenizer(f" {rec['answer']}", add_special_tokens=False)["input_ids"]
            full_ids = (prompt_ids + answer_ids)[:seq_len]
            L = len(full_ids)
            input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                _ = model(input_ids)
            upper = causal_mask(L, device)
            scale = (model.config.n_embd // model.config.n_head) ** -0.5
            for layer_idx, (_, module) in enumerate(path_layers):
                if layer_idx not in selected_layers:
                    continue
                base = getattr(module, "_last_logits_pa_only", None)
                full = getattr(module, "_last_logits_full", None)
                p_base = compute_probs(base[0, :, :L, :L], upper, scale)
                p_wav = compute_probs(full[0, :, :L, :L], upper, scale)
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
    p.add_argument("--checkpoint", default="/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat244_dual_temp/K1_L512_me16_rho256_ricker_s42/checkpoint-15000")
    p.add_argument("--seq_lens", default="512,4096")
    p.add_argument("--n_case", type=int, default=50)
    p.add_argument("--jsonl", default="data/hotpot_long_dev.jsonl")
    p.add_argument("--n_qbin", type=int, default=10)
    p.add_argument("--out_dir", default="analysis_outputs/qwab_attn_bins")
    p.add_argument("--top_k_layers", type=int, default=3)
    run(p.parse_args())


if __name__ == "__main__":
    main()

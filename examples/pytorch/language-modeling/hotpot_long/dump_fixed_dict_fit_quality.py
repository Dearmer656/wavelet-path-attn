#!/usr/bin/env python3
"""PAT-194: example head fit-quality report against the fixed train-length (L512) NMF dictionary.

For one case, fixes the global L512 NMF basis V_512 (raw, unnormalized) and
solves a per-(layer, head) NNLS fit b* = argmin_{b>=0} ||x - b V_512||^2 for
both the L512 and ext-length head maps. Reports reconstruction error, fit
cosine, top-pattern assignment, and NNLS-derived usage statistics.
"""
import argparse
import csv
import sys
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import nnls
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_layerwise_falsifiable_hypotheses import layer_bin
from analyze_pattern_basis_matching import classify_basis_rows, resolve_nmf_run_dir
from run_head_ablation_eval import (
    _capture_total_for_model,
    build_capture_input_ids,
    capture_ext_logit_maps,
    find_path_attention_modules,
    pair_cases_by_base_id,
    load_path_attn_model,
    preprocess_head_features,
)

CSV_FIELDNAMES = [
    "model", "length", "train_dictionary_length", "rank", "case_id", "layer", "head", "layer_bin",
    "recon_error", "fit_cosine", "top_pattern_id", "top_pattern_type", "top_pattern_cosine",
    "nnls_top_usage", "nnls_usage_entropy",
    "coverage_tau_010", "coverage_tau_020", "coverage_tau_030",
]


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["PaTH-only", "QWAB", "NoPE", "Rotary"], default="QWAB")
    parser.add_argument("--ext_len", type=int, choices=[2048, 4096], default=4096)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--max_cases", type=int, default=1)
    parser.add_argument("--hotpot_jsonl", type=Path, default=root / "data" / "hotpot_long_dev_uniform.jsonl")
    parser.add_argument("--out_dir", type=Path, default=root / "analysis_outputs" / "pat187_fit_quality")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def fit_against_dictionary(x: np.ndarray, V: np.ndarray, eps: float = 1e-8) -> dict:
    """Solve b* = argmin_{b>=0} ||x - bV||^2 and report fit-quality stats."""
    b, _ = nnls(V.T, x)
    recon = b @ V
    x_norm = float(np.linalg.norm(x))
    recon_norm = float(np.linalg.norm(recon))
    recon_error = float(np.linalg.norm(x - recon) / (x_norm + eps))
    fit_cosine = float((x @ recon) / (x_norm * recon_norm + eps))
    r_star = int(np.argmax(b))
    v_star = V[r_star]
    top_pattern_cosine = float((x @ v_star) / (x_norm * np.linalg.norm(v_star) + eps))
    p = b / (b.sum() + eps)
    top_usage = float(p.max())
    usage_entropy = float(-(p * np.log(p + eps)).sum())
    return {
        "b": b,
        "recon": recon,
        "recon_error": recon_error,
        "fit_cosine": fit_cosine,
        "top_pattern_id": r_star,
        "top_pattern_cosine": top_pattern_cosine,
        "top_usage": top_usage,
        "usage_entropy": usage_entropy,
    }


def capture_case_rows(model, path_modules, tokenizer, device, model_label, rec, length,
                       pool_size, preprocessing, V512, pattern_types, base_id, head_dim_scale):
    capture_total = _capture_total_for_model(model_label)
    input_ids = build_capture_input_ids(tokenizer, rec, length, device)
    logit_maps = capture_ext_logit_maps(model, path_modules, input_ids, capture_total, head_dim_scale)
    head_features, valid_heads = preprocess_head_features(logit_maps, pool_size, preprocessing)

    rows, fits, feats = [], {}, {}
    for head in valid_heads:
        x = head_features[head]
        fit = fit_against_dictionary(x, V512)
        rows.append({
            "model": model_label,
            "length": length,
            "train_dictionary_length": 512,
            "rank": V512.shape[0],
            "case_id": base_id,
            "layer": head.layer_idx,
            "head": head.head_idx,
            "layer_bin": layer_bin(head.layer_idx),
            "recon_error": fit["recon_error"],
            "fit_cosine": fit["fit_cosine"],
            "top_pattern_id": fit["top_pattern_id"],
            "top_pattern_type": pattern_types[fit["top_pattern_id"]],
            "top_pattern_cosine": fit["top_pattern_cosine"],
            "nnls_top_usage": fit["top_usage"],
            "nnls_usage_entropy": fit["usage_entropy"],
            "coverage_tau_010": int(fit["recon_error"] < 0.10),
            "coverage_tau_020": int(fit["recon_error"] < 0.20),
            "coverage_tau_030": int(fit["recon_error"] < 0.30),
        })
        fits[head] = fit
        feats[head] = x
    return rows, fits, feats


def save_head_visualization(out_path: Path, x: np.ndarray, fit: dict, V512: np.ndarray, M: int, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_2d = x.reshape(M, M)
    recon_2d = fit["recon"].reshape(M, M)
    residual_2d = x_2d - recon_2d
    v_star_2d = V512[fit["top_pattern_id"]].reshape(M, M)

    vmax = float(np.quantile(x_2d, 0.995))
    vmax = max(vmax, 1e-8)
    res_max = float(np.abs(residual_2d).max())
    res_max = max(res_max, 1e-8)
    v_vmax = float(np.quantile(v_star_2d, 0.995))
    v_vmax = max(v_vmax, 1e-8)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    panels = [
        ("original x", x_2d, "viridis", 0, vmax),
        (f"fitted bV (err={fit['recon_error']:.3f})", recon_2d, "viridis", 0, vmax),
        ("residual x - bV", residual_2d, "RdBu_r", -res_max, res_max),
        (f"top V[{fit['top_pattern_id']}]", v_star_2d, "viridis", 0, v_vmax),
    ]
    for ax, (panel_title, data, cmap, vmin, vmax_) in zip(axes, panels):
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax_)
        ax.set_title(panel_title, fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=80)
    plt.close(fig)


def write_table_md(rows: list) -> str:
    header = ["layer", "head", "recon_error", "fit_cosine", "top_pattern_id", "top_pattern_type",
              "top_pattern_cosine", "top_usage", "usage_entropy"]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for r in rows:
        lines.append(
            "| {layer} | {head} | {recon_error:.4f} | {fit_cosine:.4f} | {top_pattern_id} | "
            "{top_pattern_type} | {top_pattern_cosine:.4f} | {nnls_top_usage:.4f} | {nnls_usage_entropy:.4f} |"
            .format(**r)
        )
    return "\n".join(lines)


def main():
    args = parse_args()
    runs_root = Path(__file__).resolve().parent.parent / "runs"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir = args.out_dir / "heatmaps"
    heatmap_dir.mkdir(exist_ok=True)

    train_dir = resolve_nmf_run_dir(args.model, 512, args.rank, runs_root)
    V512 = np.load(train_dir / "V.npy").astype(np.float64, copy=False)
    with open(train_dir / "meta.json") as f:
        train_meta = json.load(f)
    pool_size = int(train_meta["pool_size"])
    preprocessing = train_meta["preprocessing"]
    pattern_types = classify_basis_rows(V512, pool_size)
    print(f"V_512: shape={V512.shape}, pattern_types={pattern_types}")

    pairs = pair_cases_by_base_id(args.hotpot_jsonl, args.ext_len, args.max_cases)
    print(f"Case pairs: {len(pairs)}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoint = train_meta["checkpoint"]
    print(f"Checkpoint: {checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = load_path_attn_model(checkpoint, device)
    path_modules = find_path_attention_modules(model)
    head_dim = model.config.n_embd // model.config.n_head
    head_dim_scale = head_dim ** -0.5

    all_rows, ext_fits, ext_feats = [], {}, {}
    for case_pair in pairs:
        for length, rec in ((512, case_pair.base_rec), (args.ext_len, case_pair.ext_rec)):
            print(f"  case {case_pair.base_id} | length {length} | capturing + fitting ...")
            rows, fits, feats = capture_case_rows(
                model, path_modules, tokenizer, device, args.model, rec, length,
                pool_size, preprocessing, V512, pattern_types, case_pair.base_id, head_dim_scale,
            )
            all_rows.extend(rows)
            if length == args.ext_len:
                ext_fits, ext_feats = fits, feats

    csv_path = args.out_dir / "example_head_fit_quality.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Saved {csv_path} ({len(all_rows)} rows)")

    base_id = pairs[0].base_id
    ext_rows = [r for r in all_rows if r["length"] == args.ext_len]
    base_rows = [r for r in all_rows if r["length"] == 512]

    ext_sorted = sorted(ext_rows, key=lambda r: r["recon_error"])
    best5 = ext_sorted[:5]
    worst5 = ext_sorted[-5:][::-1]

    # 4-panel visualizations for top-2 best-fit and top-2 worst-fit heads
    viz_entries = []
    for tag, r in [("best", best5[0]), ("best", best5[1]), ("worst", worst5[0]), ("worst", worst5[1])]:
        head_key = next(h for h in ext_fits if h.layer_idx == r["layer"] and h.head_idx == r["head"])
        fname = f"{base_id}_L{args.ext_len}_layer{r['layer']:02d}_head{r['head']:02d}_{tag}.png"
        out_path = heatmap_dir / fname
        title = (f"{args.model} L{args.ext_len} case={base_id} layer={r['layer']} head={r['head']} "
                 f"({tag}-fit, recon_error={r['recon_error']:.4f})")
        save_head_visualization(out_path, ext_feats[head_key], ext_fits[head_key], V512, pool_size, title)
        viz_entries.append((tag, r, fname))
        print(f"  saved {out_path}")

    recon_errors = np.array([r["recon_error"] for r in ext_rows])
    p10, p90 = float(np.percentile(recon_errors, 10)), float(np.percentile(recon_errors, 90))
    spread = p90 - p10
    if spread >= 0.15:
        decision, decision_reason = "proceed", f"P90-P10 spread = {spread:.4f} >= 0.15 (strong signal)"
    elif spread >= 0.10:
        decision, decision_reason = "proceed", f"P90-P10 spread = {spread:.4f} >= 0.10"
    elif spread <= 0.05:
        decision, decision_reason = "do not proceed", f"P90-P10 spread = {spread:.4f} <= 0.05 (uniform across heads)"
    else:
        decision, decision_reason = "borderline", f"P90-P10 spread = {spread:.4f} (between 0.05 and 0.10)"

    # poor-fit heads by layer_bin
    by_bin_worst = {}
    for r in worst5:
        by_bin_worst.setdefault(r["layer_bin"], 0)
        by_bin_worst[r["layer_bin"]] += 1

    base_recon_errors = np.array([r["recon_error"] for r in base_rows])

    md = []
    md.append("# PAT-194: Example Head Fit-Quality Report\n")
    md.append(f"- **Model**: {args.model}")
    md.append(f"- **Train dictionary**: V_512 (R={args.rank}, raw/unnormalized, `{train_dir}`)")
    md.append(f"- **Checkpoint**: `{checkpoint}`")
    md.append(f"- **Case** (`case_id` = HotpotQA `base_id`): `{base_id}`")
    md.append(f"- **Eval lengths**: 512 (n={len(base_rows)}), {args.ext_len} (n={len(ext_rows)})")
    md.append(f"- **Top-pattern types over V_512 rows** (`classify_basis_rows`): {pattern_types}\n")

    md.append(f"## L{args.ext_len} fit-quality table (12 layers x 12 heads, n={len(ext_rows)})\n")
    md.append(write_table_md(sorted(ext_rows, key=lambda r: (r["layer"], r["head"]))))
    md.append("")

    md.append("## Top-5 best-fit heads (lowest recon_error)\n")
    md.append(write_table_md(best5))
    md.append("")

    md.append("## Top-5 worst-fit heads (highest recon_error)\n")
    md.append(write_table_md(worst5))
    md.append("")

    md.append("## Visualizations (original x | fitted bV | residual | top V pattern)\n")
    for tag, r, fname in viz_entries:
        md.append(f"- **{tag}-fit** layer={r['layer']} head={r['head']} "
                   f"(recon_error={r['recon_error']:.4f}, top_pattern_id={r['top_pattern_id']}, "
                   f"top_pattern_type={r['top_pattern_type']}): `heatmaps/{fname}`")
        md.append(f"\n![{fname}](heatmaps/{fname})\n")

    md.append(f"## L512 sanity check (V_512 fit on its own training length, n={len(base_rows)})\n")
    md.append(f"- mean recon_error = {base_recon_errors.mean():.4f}")
    md.append(f"- median recon_error = {np.median(base_recon_errors):.4f}")
    md.append(f"- coverage@0.10 = {np.mean(base_recon_errors < 0.10):.3f}, "
              f"coverage@0.20 = {np.mean(base_recon_errors < 0.20):.3f}, "
              f"coverage@0.30 = {np.mean(base_recon_errors < 0.30):.3f}\n")

    md.append("## Interpretation\n")
    ext_mean = recon_errors.mean()
    ext_median = float(np.median(recon_errors))
    cov10 = float(np.mean(recon_errors < 0.10))
    cov20 = float(np.mean(recon_errors < 0.20))
    cov30 = float(np.mean(recon_errors < 0.30))
    md.append(
        f"- **Are most heads well represented?** At L{args.ext_len}, mean recon_error = {ext_mean:.4f}, "
        f"median = {ext_median:.4f}. Coverage@0.10 = {cov10:.3f}, @0.20 = {cov20:.3f}, @0.30 = {cov30:.3f}."
    )
    worst_bins = ", ".join(f"{k}={v}" for k, v in sorted(by_bin_worst.items()))
    md.append(
        f"- **Are poor-fit heads concentrated in specific layers?** Among the 5 worst-fit heads, "
        f"layer_bin counts: {worst_bins}. Worst-fit layers: "
        + ", ".join(f"L{r['layer']}H{r['head']}({r['layer_bin']})" for r in worst5) + "."
    )
    md.append(
        "- **Do poor-fit heads visually look semantic / retrieval-like / noisy / OOD?** "
        "See visualizations above for the worst-fit heads; compare residual magnitude and structure "
        "against the assigned top-V pattern (`top_pattern_type`)."
    )
    md.append("")

    md.append("## Go / No-Go\n")
    md.append(f"- P10(recon_error) = {p10:.4f}, P90(recon_error) = {p90:.4f}, spread = {spread:.4f}")
    md.append(f"- **Decision: {decision}** — {decision_reason}")

    report_path = args.out_dir / "example_head_fit_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"Saved {report_path}")
    print(f"Decision: {decision} ({decision_reason})")


if __name__ == "__main__":
    main()

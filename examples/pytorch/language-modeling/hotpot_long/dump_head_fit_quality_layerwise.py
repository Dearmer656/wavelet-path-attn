#!/usr/bin/env python3
"""PAT-194: per-case head fit-quality table against PER-LAYER (not global) L512 NMF
dictionaries V_layer[0..11], for the same L4096 / max_cases=20 case set used by
dump_head_fit_quality.py.

Also fits the same L{ext_len} attention-logit features against the per-layer
V_layer dictionary fit independently ON L{ext_len} itself ("extdict"/oracle fit),
so the train-dict (V_512, extrapolation) recon_error can be compared against the
ext-dict (V_{ext_len}, in-distribution) recon_error per layer.

Also produces, for one example case, a per-layer "low vs high recon_error head"
attention-logit comparison: for each layer, the head with the lowest recon_error
(best fit to the train-length motif basis for that layer) vs the head with the
highest recon_error (worst fit), visualized side-by-side (original | fitted bV |
residual | top-V pattern).

Reads:
  - runs/.../L512_checkpoint-15900_salient_R{rank}[_total]_layerwise/layer_XX/{V.npy,meta.json}
  - runs/.../L{ext_len}_checkpoint-15900_salient_R{rank}[_total]_layerwise/layer_XX/{V.npy,meta.json}

Writes:
  - analysis_outputs/pat187_fit_quality/head_fit_quality_layerwise_{model_tag}.csv
  - analysis_outputs/pat187_fit_quality/head_fit_quality_layerwise_extdict_{model_tag}.csv
  - analysis_outputs/pat187_fit_quality/head_low_high_comparison_layerwise_{model_tag}.csv
  - analysis_outputs/pat187_fit_quality/heatmaps_layerwise/{base_id}_L{ext_len}_layer{XX}_lowhigh_{model_tag}.png
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_layerwise_falsifiable_hypotheses import layer_bin
from analyze_pattern_basis_matching import classify_basis_rows
from analyze_pattern_basis_matching_layerwise import resolve_layerwise_root
from dump_fixed_dict_fit_quality import CSV_FIELDNAMES, fit_against_dictionary
from run_head_ablation_eval import (
    _capture_total_for_model,
    build_capture_input_ids,
    capture_ext_logit_maps,
    find_path_attention_modules,
    load_path_attn_model,
    pair_cases_by_base_id,
    preprocess_head_features,
)

COMPARISON_FIELDNAMES = [
    "model", "length", "case_id", "layer", "layer_bin",
    "low_head", "low_recon_error", "low_top_pattern_id", "low_top_pattern_type",
    "high_head", "high_recon_error", "high_top_pattern_id", "high_top_pattern_type",
    "recon_error_gap", "cosine_low_high_logits",
]


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["PaTH-only", "QWAB", "NoPE", "Rotary", "PA-only-M", "QWAB-M", "NoPE-M", "WPE-M"], default="QWAB")
    parser.add_argument("--ext_len", type=int, choices=[2048, 4096], default=4096)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--max_cases", type=int, default=20)
    parser.add_argument("--hotpot_jsonl", type=Path, default=root / "data" / "hotpot_long_dev_uniform.jsonl")
    parser.add_argument("--out_dir", type=Path, default=root / "analysis_outputs" / "pat187_fit_quality")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--example_case_idx", type=int, default=0,
                         help="index into the case-pair list used for the low-vs-high visualization")
    parser.add_argument("--ext_run_suffix", default="",
                         help="Suffix for the ext-length (V_{ext_len}) layerwise dict dir, e.g. "
                              "'_heldout50' to use a dictionary fit on a held-out case set as the "
                              "oracle, while the train (V_512) dict stays fixed/unsuffixed.")
    parser.add_argument("--out_suffix", default="",
                         help="Suffix appended to output CSV/PNG filenames, e.g. '_heldout50'.")
    return parser.parse_args()


def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


def fit_row(model_label, length, dict_length, base_id, layer_idx, head_idx, x, V_layer, pattern_types, n_layers=12):
    fit = fit_against_dictionary(x, V_layer)
    row = {
        "model": model_label,
        "length": length,
        "train_dictionary_length": dict_length,
        "rank": V_layer.shape[0],
        "case_id": base_id,
        "layer": layer_idx,
        "head": head_idx,
        "layer_bin": layer_bin(layer_idx, n_layers),
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
    }
    return row, fit


def capture_case_rows_layerwise(model, path_modules, tokenizer, device, model_label, rec, length,
                                 pool_size, preprocessing, V_by_layer_train, pattern_types_by_layer_train,
                                 V_by_layer_ext, pattern_types_by_layer_ext, ext_len, base_id, head_dim_scale, n_layers=12):
    capture_total = _capture_total_for_model(model_label)
    input_ids = build_capture_input_ids(tokenizer, rec, length, device)
    logit_maps = capture_ext_logit_maps(model, path_modules, input_ids, capture_total, head_dim_scale)
    head_features, valid_heads = preprocess_head_features(logit_maps, pool_size, preprocessing)

    rows_train, rows_ext, fits_train, feats = [], [], {}, {}
    for head in valid_heads:
        x = head_features[head]
        row_train, fit_train = fit_row(
            model_label, length, 512, base_id, head.layer_idx, head.head_idx, x,
            V_by_layer_train[head.layer_idx], pattern_types_by_layer_train[head.layer_idx], n_layers,
        )
        row_ext, _fit_ext = fit_row(
            model_label, length, ext_len, base_id, head.layer_idx, head.head_idx, x,
            V_by_layer_ext[head.layer_idx], pattern_types_by_layer_ext[head.layer_idx], n_layers,
        )
        rows_train.append(row_train)
        rows_ext.append(row_ext)
        fits_train[head] = fit_train
        feats[head] = x
    return rows_train, rows_ext, fits_train, feats


def save_low_high_comparison(out_path, low_x, low_fit, low_V, high_x, high_fit, high_V, M, title, low_label, high_label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for row_axes, label, x, fit, V in (
        (axes[0], low_label, low_x, low_fit, low_V),
        (axes[1], high_label, high_x, high_fit, high_V),
    ):
        x_2d = x.reshape(M, M)
        recon_2d = fit["recon"].reshape(M, M)
        residual_2d = x_2d - recon_2d
        v_star_2d = V[fit["top_pattern_id"]].reshape(M, M)

        vmax = max(float(np.quantile(x_2d, 0.995)), 1e-8)
        res_max = max(float(np.abs(residual_2d).max()), 1e-8)
        v_vmax = max(float(np.quantile(v_star_2d, 0.995)), 1e-8)

        panels = [
            (f"{label}: original x", x_2d, "viridis", 0, vmax),
            (f"fitted bV (err={fit['recon_error']:.3f})", recon_2d, "viridis", 0, vmax),
            ("residual x - bV", residual_2d, "RdBu_r", -res_max, res_max),
            (f"top V[{fit['top_pattern_id']}]", v_star_2d, "viridis", 0, v_vmax),
        ]
        for ax, (panel_title, data, cmap, vmin, vmax_) in zip(row_axes, panels):
            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax_)
            ax.set_title(panel_title, fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=80)
    plt.close(fig)


def main():
    args = parse_args()
    runs_root = Path(__file__).resolve().parent.parent / "runs"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir = args.out_dir / "heatmaps_layerwise"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    layerwise_dir = resolve_layerwise_root(args.model, 512, args.rank, runs_root)
    ext_layerwise_dir = resolve_layerwise_root(args.model, args.ext_len, args.rank, runs_root, suffix=args.ext_run_suffix)
    with open(layerwise_dir / "layer_00" / "meta.json") as f:
        layer0_meta = json.load(f)
    N_LAYERS = int(layer0_meta["n_layers"])
    pool_size = int(layer0_meta["pool_size"])
    preprocessing = layer0_meta["preprocessing"]
    checkpoint = layer0_meta["checkpoint"]

    def load_layerwise_V(layerwise_root):
        V_by_layer, pattern_types_by_layer = {}, {}
        for layer_idx in range(N_LAYERS):
            V = np.load(layerwise_root / f"layer_{layer_idx:02d}" / "V.npy").astype(np.float64, copy=False)
            V_by_layer[layer_idx] = V
            pattern_types_by_layer[layer_idx] = classify_basis_rows(V, pool_size)
        return V_by_layer, pattern_types_by_layer

    V_by_layer, pattern_types_by_layer = load_layerwise_V(layerwise_dir)
    print(f"Loaded {N_LAYERS} layer-wise V_512 (train) dictionaries from {layerwise_dir}")
    for layer_idx in range(N_LAYERS):
        print(f"  layer {layer_idx:02d}: shape={V_by_layer[layer_idx].shape}, pattern_types={pattern_types_by_layer[layer_idx]}")

    V_by_layer_ext, pattern_types_by_layer_ext = load_layerwise_V(ext_layerwise_dir)
    print(f"Loaded {N_LAYERS} layer-wise V_{args.ext_len} (ext, oracle) dictionaries from {ext_layerwise_dir}")
    for layer_idx in range(N_LAYERS):
        print(f"  layer {layer_idx:02d}: shape={V_by_layer_ext[layer_idx].shape}, pattern_types={pattern_types_by_layer_ext[layer_idx]}")

    pairs = pair_cases_by_base_id(args.hotpot_jsonl, args.ext_len, args.max_cases)
    print(f"Case pairs: {len(pairs)}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Checkpoint: {checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = load_path_attn_model(checkpoint, device)
    path_modules = find_path_attention_modules(model)
    head_dim = model.config.n_embd // model.config.n_head
    head_dim_scale = head_dim ** -0.5

    all_rows, all_rows_ext = [], []
    feats_by_key, fits_by_key = {}, {}
    rows_by_case_layer = {}
    for i, case_pair in enumerate(pairs, start=1):
        print(f"  case {i}/{len(pairs)}: {case_pair.base_id} | length {args.ext_len} | capturing + fitting (layer-wise V) ...")
        rows, rows_ext, fits, feats = capture_case_rows_layerwise(
            model, path_modules, tokenizer, device, args.model, case_pair.ext_rec, args.ext_len,
            pool_size, preprocessing, V_by_layer, pattern_types_by_layer,
            V_by_layer_ext, pattern_types_by_layer_ext, args.ext_len, case_pair.base_id, head_dim_scale, N_LAYERS,
        )
        all_rows.extend(rows)
        all_rows_ext.extend(rows_ext)
        for row in rows:
            rows_by_case_layer.setdefault((row["case_id"], row["layer"]), []).append(row)
        for head, fit in fits.items():
            key = (case_pair.base_id, head.layer_idx, head.head_idx)
            feats_by_key[key] = feats[head]
            fits_by_key[key] = fit

    model_tag = args.model.replace("-", "").lower() + args.out_suffix
    csv_path = args.out_dir / f"head_fit_quality_layerwise_{model_tag}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Saved {csv_path} ({len(all_rows)} rows)")

    extdict_csv_path = args.out_dir / f"head_fit_quality_layerwise_extdict_{model_tag}.csv"
    with open(extdict_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows_ext)
    print(f"Saved {extdict_csv_path} ({len(all_rows_ext)} rows)")

    # Per (case, layer): low-recon-error head vs high-recon-error head, comparing
    # their raw attention-logit maps.
    comparison_rows = []
    for (case_id, layer_idx), layer_rows in rows_by_case_layer.items():
        low = min(layer_rows, key=lambda r: r["recon_error"])
        high = max(layer_rows, key=lambda r: r["recon_error"])
        x_low = feats_by_key[(case_id, layer_idx, low["head"])]
        x_high = feats_by_key[(case_id, layer_idx, high["head"])]
        comparison_rows.append({
            "model": args.model,
            "length": args.ext_len,
            "case_id": case_id,
            "layer": layer_idx,
            "layer_bin": layer_bin(layer_idx),
            "low_head": low["head"],
            "low_recon_error": low["recon_error"],
            "low_top_pattern_id": low["top_pattern_id"],
            "low_top_pattern_type": low["top_pattern_type"],
            "high_head": high["head"],
            "high_recon_error": high["recon_error"],
            "high_top_pattern_id": high["top_pattern_id"],
            "high_top_pattern_type": high["top_pattern_type"],
            "recon_error_gap": high["recon_error"] - low["recon_error"],
            "cosine_low_high_logits": cosine_sim(x_low, x_high),
        })
    comparison_rows.sort(key=lambda r: (r["case_id"], r["layer"]))
    comp_csv_path = args.out_dir / f"head_low_high_comparison_layerwise_{model_tag}.csv"
    with open(comp_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COMPARISON_FIELDNAMES)
        writer.writeheader()
        writer.writerows(comparison_rows)
    print(f"Saved {comp_csv_path} ({len(comparison_rows)} rows)")

    # Example-case visualizations: for each layer, low-vs-high recon_error head.
    example_case = pairs[args.example_case_idx].base_id
    for layer_idx in range(N_LAYERS):
        comp = next(c for c in comparison_rows if c["case_id"] == example_case and c["layer"] == layer_idx)
        low_key = (example_case, layer_idx, comp["low_head"])
        high_key = (example_case, layer_idx, comp["high_head"])
        fname = f"{example_case}_L{args.ext_len}_layer{layer_idx:02d}_lowhigh_{model_tag}.png"
        out_path = heatmap_dir / fname
        title = (f"{args.model} L{args.ext_len} case={example_case} layer={layer_idx} (layer-wise V_512): "
                 f"low recon_error head={comp['low_head']} ({comp['low_recon_error']:.4f}) vs "
                 f"high recon_error head={comp['high_head']} ({comp['high_recon_error']:.4f}), "
                 f"cosine(x_low,x_high)={comp['cosine_low_high_logits']:.4f}")
        save_low_high_comparison(
            out_path,
            feats_by_key[low_key], fits_by_key[low_key], V_by_layer[layer_idx],
            feats_by_key[high_key], fits_by_key[high_key], V_by_layer[layer_idx],
            pool_size, title,
            f"low (head {comp['low_head']})", f"high (head {comp['high_head']})",
        )
        print(f"  saved {out_path}")

    recon_errors = np.array([r["recon_error"] for r in all_rows])
    print(f"V_512 (extrapolation): n={len(all_rows)}, mean recon_error={recon_errors.mean():.4f}, "
          f"P25={np.percentile(recon_errors, 25):.4f}, P50={np.percentile(recon_errors, 50):.4f}, "
          f"P75={np.percentile(recon_errors, 75):.4f}")
    recon_errors_ext = np.array([r["recon_error"] for r in all_rows_ext])
    print(f"V_{args.ext_len} (oracle): n={len(all_rows_ext)}, mean recon_error={recon_errors_ext.mean():.4f}, "
          f"P25={np.percentile(recon_errors_ext, 25):.4f}, P50={np.percentile(recon_errors_ext, 50):.4f}, "
          f"P75={np.percentile(recon_errors_ext, 75):.4f}")
    gaps = np.array([r["recon_error_gap"] for r in comparison_rows])
    coss = np.array([r["cosine_low_high_logits"] for r in comparison_rows])
    print(f"low-vs-high: n={len(comparison_rows)}, mean recon_error_gap={gaps.mean():.4f}, "
          f"mean cosine(x_low,x_high)={coss.mean():.4f}")


if __name__ == "__main__":
    main()

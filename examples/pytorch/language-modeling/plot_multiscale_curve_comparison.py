#!/usr/bin/env python
"""Plot the actual bias-vs-key-position curve for K1/K2/K4/K5/K6, overlaid.

For each K-value model, at a fixed checkpoint and the LAST query position of
a fixed eval_length (so the full causal span k=0..T-1 is visible, showing
the complete multi-scale wavelet shape anchored near k=0), this:

  1. Reads each scale's RMS-normalized + p99-clamped basis (S2_postp99,
     the same "basis_s" used everywhere else in this analysis suite).
  2. Sums the K scales UNWEIGHTED (equal weight, no router pi_scale gating)
     -- "each scale RMS'd first, then summed" -- to isolate the raw
     multi-scale shape from whatever the learned router currently prefers.
  3. Averages this row over samples and layers to get one curve per model.
  4. Centers the curve (subtracts its own mean, matching the
     causal-centered convention used throughout this analysis suite --
     softmax ignores a constant per-row shift).
  5. Marks the peak (max) and valley (min) point on each curve.
  6. Plots all K-value curves on one figure.

Does not modify fla/layers/path_attn.py; reuses model-loading and capture
helpers from analyze_k1_k4_wavelet_amp.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from analyze_k1_k4_wavelet_amp import (
    build_eval_samples,
    discover_checkpoints,
    load_model_for_analysis,
    _clear_capture,
    _set_capture,
)

DEFAULT_RUNS: Dict[str, Dict[str, str]] = {
    "K1": {
        "dir": "K1_me16_noC1_s42",
        "label": "K1 (with_null)",
    },
    "K2": {
        "dir": "K2_me16_24_noC1_s42_withnull_sqrtnorm_ablation",
        "label": "K2 (with_null)",
    },
    "K4": {
        "dir": "K4_me8_16_20_24_noC1_s42_sqrtnorm",
        "label": "K4 (with_null)",
    },
    "K5": {
        "dir": "K5_me8_12_16_20_24_noC1_s42_independent_sqrtnorm",
        "label": "K5 (with_null_independent_scales)",
    },
    "K6": {
        "dir": "K6_me8_12_14_16_20_24_noC1_s42_independent_sqrtnorm",
        "label": "K6 (with_null_independent_scales)",
    },
}


def capture_unweighted_curve_for_last_query(loaded, samples: torch.Tensor, device: str, batch_size: int) -> np.ndarray:
    """Returns a 1D array of length T: the unweighted sum-of-RMS-normalized-scales
    curve at the LAST query position, averaged over samples and layers."""
    t = int(samples.shape[1])
    per_batch_layer_curves: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(samples), int(batch_size)):
            input_ids = samples[start : start + int(batch_size)].to(device)
            _set_capture(loaded.layers, True)
            loaded.model(input_ids=input_ids, labels=None, use_cache=False)
            for layer in loaded.layers:
                cap = getattr(layer, "_pat234_cap", None)
                if cap is None:
                    raise RuntimeError("Layer capture dict is not set")
                s2 = cap.get("S2_postp99", [])
                if not s2:
                    raise RuntimeError(f"Missing S2_postp99 for layer {getattr(layer, 'layer_idx', '?')}")
                by_q0: Dict[int, Dict[int, torch.Tensor]] = {}
                for _lid, scale_idx, q0, basis in s2:
                    by_q0.setdefault(int(q0), {})[int(scale_idx)] = basis.float()
                last_q0 = max(by_q0)
                scales_at_last = by_q0[last_q0]
                k_expected = int(getattr(layer, "wavelet_ctxscale_k"))
                if len(scales_at_last) != k_expected:
                    raise RuntimeError(
                        f"layer {getattr(layer, 'layer_idx', '?')}: expected {k_expected} scales at q0={last_q0}, got {len(scales_at_last)}"
                    )
                # basis_i shape [B, q_chunk, T]; unweighted sum across scales.
                stacked = torch.stack([scales_at_last[i] for i in sorted(scales_at_last)], dim=0)  # [K, B, q_chunk, T]
                unweighted_sum = stacked.sum(dim=0)  # [B, q_chunk, T]
                last_row_in_chunk = unweighted_sum.shape[1] - 1
                # Confirm this chunk really contains the last absolute query t-1.
                assert last_q0 + last_row_in_chunk == t - 1, (
                    f"last captured chunk q0={last_q0} + row {last_row_in_chunk} != t-1={t - 1}"
                )
                curve = unweighted_sum[:, last_row_in_chunk, :]  # [B, T]
                per_batch_layer_curves.append(curve.numpy())
            _clear_capture(loaded.layers)
    all_curves = np.concatenate(per_batch_layer_curves, axis=0)  # [num_layers * num_samples, T]
    return all_curves.mean(axis=0)  # [T]


def causal_center_full_row(curve: np.ndarray) -> np.ndarray:
    """Row-mean-center a full-length curve (the last query sees the whole causal span,
    so this is just a plain mean subtraction -- softmax ignores a constant shift)."""
    return curve - curve.mean()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_dir", default=str(Path(__file__).parent / "runs" / "pat234_scale_card"))
    p.add_argument("--checkpoint_step", type=int, default=15000)
    p.add_argument("--eval_length", type=int, default=512)
    p.add_argument("--num_samples", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["auto", "float32", "float16", "bfloat16"])
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--output_dir", default="analysis_outputs/multiscale_curve_comparison/")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_dir = Path(args.base_dir)

    curves: Dict[str, np.ndarray] = {}
    labels: Dict[str, str] = {}
    manifest: Dict[str, Any] = {"checkpoint_step": args.checkpoint_step, "eval_length": args.eval_length, "num_samples": args.num_samples, "runs": {}}
    sample_cache = None

    for k_name, info in DEFAULT_RUNS.items():
        run_dir = base_dir / info["dir"]
        ckpts = discover_checkpoints(run_dir)
        if args.checkpoint_step not in ckpts:
            raise RuntimeError(f"{k_name}: checkpoint-{args.checkpoint_step} not found under {run_dir}")
        loaded = load_model_for_analysis(ckpts[args.checkpoint_step], run_dir, args.device, args.dtype)
        if sample_cache is None:
            sample_cache = build_eval_samples(
                loaded.tokenizer, int(args.eval_length), int(args.num_samples), int(args.seed), cache_dir=args.cache_dir
            )
        raw_curve = capture_unweighted_curve_for_last_query(loaded, sample_cache, args.device, args.batch_size)
        centered = causal_center_full_row(raw_curve)
        curves[k_name] = centered
        labels[k_name] = info["label"]
        manifest["runs"][k_name] = {
            "dir": str(run_dir),
            "label": info["label"],
            "wavelet_router_sigmoid_mode": str(getattr(loaded.layers[0], "wavelet_router_sigmoid_mode", "")),
            "wavelet_ctxscale_k": int(getattr(loaded.layers[0], "wavelet_ctxscale_k")),
            "peak": float(centered.max()),
            "peak_pos": int(centered.argmax()),
            "valley": float(centered.min()),
            "valley_pos": int(centered.argmin()),
        }
        print(f"{k_name}: peak={centered.max():.4f} at k={centered.argmax()}, valley={centered.min():.4f} at k={centered.argmin()}")
        del loaded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    np.savez(output_dir / "curves.npz", **curves)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = plt.cm.tab10.colors
    for i, (k_name, curve) in enumerate(curves.items()):
        x = np.arange(len(curve))
        color = colors[i % len(colors)]
        ax.plot(x, curve, label=labels[k_name], color=color, linewidth=1.5)
        peak_pos = int(curve.argmax())
        valley_pos = int(curve.argmin())
        ax.scatter([peak_pos], [curve[peak_pos]], marker="^", s=90, color=color, edgecolor="black", zorder=5)
        ax.scatter([valley_pos], [curve[valley_pos]], marker="v", s=90, color=color, edgecolor="black", zorder=5)
        ax.annotate(
            f"{k_name} peak {curve[peak_pos]:.3f}",
            (peak_pos, curve[peak_pos]),
            textcoords="offset points",
            xytext=(4, 6),
            fontsize=8,
            color=color,
        )
        ax.annotate(
            f"{k_name} valley {curve[valley_pos]:.3f}",
            (valley_pos, curve[valley_pos]),
            textcoords="offset points",
            xytext=(4, -12),
            fontsize=8,
            color=color,
        )
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_xlabel("Key position k")
    ax.set_ylabel("Unweighted sum-of-RMS-normalized-scales, row-centered (last query)")
    ax.set_title(
        f"K1/K2/K4/K5/K6 multi-scale bias curve at checkpoint-{args.checkpoint_step}, "
        f"eval_length={args.eval_length}, query=last (n={args.num_samples})"
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "multiscale_curve_comparison.png", dpi=150)
    plt.close(fig)

    print(f"Done. Outputs written under {output_dir}")


if __name__ == "__main__":
    main()

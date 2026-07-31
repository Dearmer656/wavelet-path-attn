#!/usr/bin/env python
"""Pure analytic (no checkpoint loading) plot of context-length-RMS-normalized
Ricker wavelet curves for K1, and adjustable-weight sums for K2/K4/K5.

Uses the exact same formulas as fla/layers/path_attn.py, but computed
directly from scale values (rho = 2^(me/2), the PAT-225 fixed-support grid),
with the fixed beta=0 absolute-anchor convention (matching the "center0"
runs analyzed elsewhere in this project) -- no model weights, no torch, no
GPU needed.

  ricker(u)            = (1 - u^2) * exp(-u^2/2)                  [path_attn.py:5838-5839]
  basis_s(k)            = ricker((k - beta) / s), beta = 0
  context-length RMS     = basis_s / sqrt(mean_k(basis_s(k)^2) + eps)   [path_attn.py:5822-5825, mask=None -> full T]

K1: one standalone (single scale, unweighted) curve PER SCALE THAT K5
INCLUDES (me=8,12,16,20,24) -- not just a single me=16 reference -- so each
K5 component scale has its own K1-equivalent curve to compare against.
K2/K4/K5: each scale's normalized curve summed with an ADJUSTABLE weight per
scale (--k2_weights / --k4_weights / --k5_weights; default: all-ones, i.e.
equal/unweighted, matching the earlier unweighted-sum comparison -- change
these to see how re-weighting the scales reshapes the combined curve).

Peak and valley of each (row-mean-centered, matching the causal-centered
convention used throughout this analysis suite -- softmax ignores a
constant per-row shift) curve are marked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

ME_GRID: Dict[str, List[int]] = {
    "K2": [16, 24],
    "K4": [8, 16, 20, 24],
    "K5": [8, 12, 16, 20, 24],
}
# K1 is plotted once per scale that K5 includes (not just a single me=16
# reference), so each K5 component scale has its own standalone K1-style
# (single scale, unweighted) curve to compare against.
K1_ME_VALUES: List[int] = ME_GRID["K5"]


def me_to_rho(me: int) -> float:
    return float(2.0 ** (me / 2.0))


def ricker(u: np.ndarray) -> np.ndarray:
    return (1.0 - u**2) * np.exp(-0.5 * u**2)


def context_length_rms_normalize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Divide by sqrt(mean(x^2) + eps) over the FULL length (context-length RMS,
    matching fla's _rms_norm_last_dim with mask=None -- non-causal, position-independent)."""
    denom = np.sqrt(np.mean(x**2) + eps)
    return x / denom


def basis_curve(rho: float, t: int, beta: float = 0.0) -> np.ndarray:
    k = np.arange(t, dtype=np.float64)
    u = (k - beta) / rho
    raw = ricker(u)
    return context_length_rms_normalize(raw)


def weighted_sum_curve(rhos: Sequence[float], weights: Sequence[float], t: int, beta: float = 0.0) -> np.ndarray:
    if len(rhos) != len(weights):
        raise ValueError(f"got {len(rhos)} scales but {len(weights)} weights")
    curves = [basis_curve(rho, t, beta) for rho in rhos]
    return sum(w * c for w, c in zip(weights, curves))


def center(curve: np.ndarray) -> np.ndarray:
    return curve - curve.mean()


def parse_weights(arg: List[float] | None, k: int, name: str) -> List[float]:
    if arg is None:
        return [1.0] * k
    if len(arg) != k:
        raise ValueError(f"--{name}_weights expects {k} values (one per scale), got {len(arg)}: {arg}")
    return list(arg)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--t", type=int, default=512, help="Context length (number of key positions)")
    p.add_argument("--beta", type=float, default=0.0, help="Fixed anchor position (absolute-center convention)")
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--k2_weights", type=float, nargs="+", default=None, help="2 weights, one per K2 scale [me=16,24]")
    p.add_argument("--k4_weights", type=float, nargs="+", default=None, help="4 weights, one per K4 scale [me=8,16,20,24]")
    p.add_argument("--k5_weights", type=float, nargs="+", default=None, help="5 weights, one per K5 scale [me=8,12,16,20,24]")
    p.add_argument("--output_dir", default="analysis_outputs/analytic_multiscale_curves/")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    k2_w = parse_weights(args.k2_weights, len(ME_GRID["K2"]), "k2")
    k4_w = parse_weights(args.k4_weights, len(ME_GRID["K4"]), "k4")
    k5_w = parse_weights(args.k5_weights, len(ME_GRID["K5"]), "k5")

    rhos = {name: [me_to_rho(me) for me in mes] for name, mes in ME_GRID.items()}

    curves: Dict[str, np.ndarray] = {}
    weights_used: Dict[str, List[float]] = {}
    me_grid_used: Dict[str, List[int]] = {}
    rho_grid_used: Dict[str, List[float]] = {}

    for me in K1_ME_VALUES:
        rho = me_to_rho(me)
        name = f"K1(me={me})"
        curves[name] = center(basis_curve(rho, args.t, args.beta))
        weights_used[name] = [1.0]
        me_grid_used[name] = [me]
        rho_grid_used[name] = [rho]

    curves["K2"] = center(weighted_sum_curve(rhos["K2"], k2_w, args.t, args.beta))
    curves["K4"] = center(weighted_sum_curve(rhos["K4"], k4_w, args.t, args.beta))
    curves["K5"] = center(weighted_sum_curve(rhos["K5"], k5_w, args.t, args.beta))
    weights_used.update({"K2": k2_w, "K4": k4_w, "K5": k5_w})
    me_grid_used.update({"K2": ME_GRID["K2"], "K4": ME_GRID["K4"], "K5": ME_GRID["K5"]})
    rho_grid_used.update({"K2": rhos["K2"], "K4": rhos["K4"], "K5": rhos["K5"]})

    manifest = {"t": args.t, "beta": args.beta, "eps": args.eps, "me_grid": me_grid_used, "rho_grid": rho_grid_used, "weights": weights_used, "peaks": {}, "valleys": {}}
    for name, curve in curves.items():
        peak_pos, valley_pos = int(curve.argmax()), int(curve.argmin())
        manifest["peaks"][name] = {"value": float(curve[peak_pos]), "k": peak_pos}
        manifest["valleys"][name] = {"value": float(curve[valley_pos]), "k": valley_pos}
        print(f"{name}: rho={rho_grid_used[name]} weights={weights_used[name]} peak={curve[peak_pos]:.4f}@k={peak_pos} valley={curve[valley_pos]:.4f}@k={valley_pos}")

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    np.savez(output_dir / "curves.npz", **curves)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = plt.cm.tab10.colors
    for i, (name, curve) in enumerate(curves.items()):
        color = colors[i % len(colors)]
        x = np.arange(len(curve))
        me_str = ",".join(str(m) for m in me_grid_used[name])
        w_str = ",".join(f"{w:g}" for w in weights_used[name])
        label = f"{name} (me=[{me_str}], w=[{w_str}])"
        ax.plot(x, curve, label=label, color=color, linewidth=1.5)
        peak_pos, valley_pos = int(curve.argmax()), int(curve.argmin())
        ax.scatter([peak_pos], [curve[peak_pos]], marker="^", s=90, color=color, edgecolor="black", zorder=5)
        ax.scatter([valley_pos], [curve[valley_pos]], marker="v", s=90, color=color, edgecolor="black", zorder=5)
        ax.annotate(f"{name} peak {curve[peak_pos]:.3f}", (peak_pos, curve[peak_pos]), textcoords="offset points", xytext=(4, 6), fontsize=8, color=color)
        ax.annotate(f"{name} valley {curve[valley_pos]:.3f}", (valley_pos, curve[valley_pos]), textcoords="offset points", xytext=(4, -12), fontsize=8, color=color)
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_xlabel("Key position k")
    ax.set_ylabel("Context-length-RMS-normalized basis, weighted sum, row-centered")
    ax.set_title(f"Analytic (no-checkpoint) K1/K2/K4/K5 multi-scale curves, T={args.t}, beta={args.beta}")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "analytic_multiscale_curves.png", dpi=150)
    plt.close(fig)

    print(f"Done. Outputs written under {output_dir}")


if __name__ == "__main__":
    main()

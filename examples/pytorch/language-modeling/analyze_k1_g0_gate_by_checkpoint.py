"""PAT-243: heatmap of K1_me16 center0's router non-null gate g0_gate(layer, position)
across training checkpoints, plus adjacent-checkpoint difference heatmaps.

K=1 with_null router: pi_scale = g0_gate * w, and w=1 exactly when K=1 (only one
non-null scale, no competition to normalize against), so g0_gate IS "scale usage"
for this experiment -- no separate quantity needed.

Uses the opt-in `_pat_g0_cap` hook added to PaTHAttention._build_ctxscale_shift_logit_bias_v0
(default off, no effect on the main training path). Reuses loading/eval-sample utilities
from analyze_k1_k4_wavelet_amp.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from analyze_k1_k4_wavelet_amp import (
    build_eval_samples,
    discover_checkpoints,
    find_path_layers,
    load_model_for_analysis,
)

CHECKPOINT_STEPS = [2500, 5000, 7500, 10000, 12500, 15000]


def collect_g0_gate(run_dir: Path, ckpt_dir: Path, device: str, dtype_name: str,
                     eval_length: int, num_samples: int, seed: int, cache_dir: str | None,
                     micro_batch_size: int = 8) -> np.ndarray:
    """Returns array of shape [num_layers, eval_length], averaged over batch."""
    loaded = load_model_for_analysis(ckpt_dir, run_dir, device=device, dtype_name=dtype_name)
    layers = loaded.layers
    num_layers = len(layers)
    for layer in layers:
        layer._pat_g0_cap = {}

    batches = build_eval_samples(
        loaded.tokenizer, eval_length=eval_length, num_samples=num_samples, seed=seed, cache_dir=cache_dir
    )
    accum = np.zeros((num_layers, eval_length), dtype=np.float64)
    count = 0
    bs = int(micro_batch_size)
    with torch.no_grad():
        for start in range(0, batches.shape[0], bs):
            chunk = batches[start:start + bs].to(device)
            for layer in layers:
                layer._pat_g0_cap = {}
            loaded.model(input_ids=chunk)
            per_layer: Dict[int, torch.Tensor] = {}
            for layer in layers:
                cap = layer._pat_g0_cap
                if cap is None or "g0_gate" not in cap:
                    raise RuntimeError(f"layer {getattr(layer, 'layer_idx', '?')} did not populate _pat_g0_cap; "
                                        "check wavelet_router_sigmoid_mode / hook wiring.")
                lid, g0 = cap["g0_gate"][-1]  # [B, T, 1], most recent forward call
                per_layer[int(lid)] = g0
            for lid in range(num_layers):
                g0 = per_layer[lid].squeeze(-1)  # [B, T]
                accum[lid] += g0.sum(dim=0).cpu().numpy()
            count += chunk.shape[0]
            for layer in layers:
                layer._pat_g0_cap = None
    accum /= max(1, count)
    del loaded
    torch.cuda.empty_cache()
    return accum  # [num_layers, T]


def plot_stage_heatmaps(matrices: Dict[int, np.ndarray], output_dir: Path) -> None:
    n = len(CHECKPOINT_STEPS)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).reshape(-1)
    vmin = min(m.min() for m in matrices.values())
    vmax = max(m.max() for m in matrices.values())
    im = None
    for i, step in enumerate(CHECKPOINT_STEPS):
        ax = axes[i]
        im = ax.imshow(matrices[step], aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"checkpoint-{step}")
        ax.set_xlabel("Length (query position)")
        ax.set_ylabel("layer")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.colorbar(im, ax=axes.tolist(), shrink=0.8, label="g0_gate (scale usage)")
    fig.suptitle("K1_me16 center0: g0_gate(layer, position) across checkpoints")
    out = output_dir / "k1_me16_g0_gate_by_checkpoint.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")


def plot_diff_heatmaps(matrices: Dict[int, np.ndarray], output_dir: Path) -> None:
    pairs = list(zip(CHECKPOINT_STEPS[:-1], CHECKPOINT_STEPS[1:]))
    diffs = [matrices[b] - matrices[a] for a, b in pairs]
    vmax = max(np.abs(d).max() for d in diffs)
    vmin = -vmax
    ncols = 3
    nrows = (len(pairs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).reshape(-1)
    im = None
    for i, ((a, b), d) in enumerate(zip(pairs, diffs)):
        ax = axes[i]
        im = ax.imshow(d, aspect="auto", origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax.set_title(f"ckpt-{b} minus ckpt-{a}")
        ax.set_xlabel("Length (query position)")
        ax.set_ylabel("layer")
    for j in range(len(pairs), len(axes)):
        axes[j].axis("off")
    fig.colorbar(im, ax=axes.tolist(), shrink=0.8, label="delta g0_gate")
    fig.suptitle("K1_me16 center0: adjacent-checkpoint delta of g0_gate(layer, position)")
    out = output_dir / "k1_me16_g0_gate_adjacent_diff.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str,
                    default="runs/pat234_scale_card/K1_me16_noC1_s42_center0_ricker_norouterrms_lang01_a100x2")
    p.add_argument("--output_dir", type=str, default="pat243_g0_gate_analysis")
    p.add_argument("--eval_length", type=int, default=512)
    p.add_argument("--num_samples", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="float32")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--micro_batch_size", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpts = discover_checkpoints(run_dir)
    missing = [s for s in CHECKPOINT_STEPS if s not in ckpts]
    if missing:
        raise RuntimeError(f"Missing checkpoints in {run_dir}: {missing}. Found: {sorted(ckpts)}")

    matrices: Dict[int, np.ndarray] = {}
    for step in CHECKPOINT_STEPS:
        print(f"[main] collecting g0_gate for checkpoint-{step} ...")
        mat = collect_g0_gate(
            run_dir=run_dir,
            ckpt_dir=ckpts[step],
            device=args.device,
            dtype_name=args.dtype,
            eval_length=args.eval_length,
            num_samples=args.num_samples,
            seed=args.seed,
            cache_dir=args.cache_dir,
            micro_batch_size=args.micro_batch_size,
        )
        matrices[step] = mat
        np.save(output_dir / f"g0_gate_ckpt{step}.npy", mat)

    plot_stage_heatmaps(matrices, output_dir)
    plot_diff_heatmaps(matrices, output_dir)
    print(f"[main] done. outputs in {output_dir}")


if __name__ == "__main__":
    main()

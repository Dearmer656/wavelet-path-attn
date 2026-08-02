"""Layer x position heatmaps of the causal-centered PEAK and VALLEY of the
EFFECTIVE wavelet bias (pi_scale * basis, post layer-gate/clamp -- i.e. what
actually gets added to the attention logits), for every completed K1 (K=1)
experiment trained after the router_logits RMS-norm removal (commit
968a971f81, 2026-07-31 09:26:58), at both L=512 (training length) and
L=4096 (extrapolated).

Reuses the existing, validated `_pat234_cap` capture machinery and
`causal_centered_peak_valley` / `collect_layer_amplitudes` from
analyze_k1_k4_wavelet_amp.py -- no new capture hooks needed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from analyze_k1_k4_wavelet_amp import (
    _clear_capture,
    _set_capture,
    build_eval_samples,
    collect_layer_amplitudes,
    discover_checkpoints,
    load_model_for_analysis,
)

# (run_dir_name, short_label)
RUN_DIRS: List[Tuple[str, str]] = [
    ("K1_me4_noC1_s42_center0_ricker_norouterrms_elm43_a100x4", "me4_center0"),
    ("K1_me8_noC1_s42_center0_ricker_norouterrms_elm73_6000x4", "me8_center0"),
    ("K1_me12_noC1_s42_center0_ricker_norouterrms_lang01_a100x2", "me12_center0"),
    ("K1_me16_noC1_s42_center0_ricker_norouterrms_lang01_a100x2", "me16_center0"),
    ("K1_me16_noC1_s42_center10_ricker_norouterrms_lang01_a100x2", "me16_center_at_query"),
    ("K1_me16_noC1_s42_dual_center_seprms_sqrt2_norouterrms", "me16_dual_seprms"),
    ("K1_me16_noC1_s42_dual_center_sumrms", "me16_dual_sumrms"),
    ("K1_me20_noC1_s42_center0_ricker_norouterrms_elm73_6000x4", "me20_center0"),
    ("K1_me24_noC1_s42_center0_ricker_norouterrms_elm73_6000x4", "me24_center0"),
]


def collect_peak_valley(run_dir: Path, ckpt_dir: Path, device: str, dtype_name: str,
                         eval_length: int, num_samples: int, seed: int, cache_dir: str | None,
                         micro_batch_size: int, restore_router_rms: bool = False) -> Dict[str, np.ndarray]:
    loaded = load_model_for_analysis(ckpt_dir, run_dir, device=device, dtype_name=dtype_name)
    layers = loaded.layers
    num_layers = len(layers)
    if restore_router_rms:
        for layer in layers:
            layer._pat_restore_router_logits_rms = True

    batches = build_eval_samples(
        loaded.tokenizer, eval_length=eval_length, num_samples=num_samples, seed=seed, cache_dir=cache_dir
    )
    sum_peak = np.zeros((num_layers, eval_length), dtype=np.float64)
    sum_valley = np.zeros((num_layers, eval_length), dtype=np.float64)
    count = np.zeros((num_layers, eval_length), dtype=np.float64)
    bs = int(micro_batch_size)
    with torch.no_grad():
        for start in range(0, batches.shape[0], bs):
            input_ids = batches[start:start + bs].to(device)
            _set_capture(layers, True)
            loaded.model(input_ids=input_ids, labels=None, use_cache=False)
            for layer in layers:
                lid = int(getattr(layer, "layer_idx", 0) or 0)
                rows = collect_layer_amplitudes(layer, t=input_ids.shape[1])
                for r in rows:
                    q0 = int(r["q0"])
                    peak = np.asarray(r["effective_peak"], dtype=np.float64)
                    valley = np.asarray(r["effective_valley"], dtype=np.float64)
                    qlen = peak.shape[0]
                    sum_peak[lid, q0:q0 + qlen] += peak
                    sum_valley[lid, q0:q0 + qlen] += valley
                    count[lid, q0:q0 + qlen] += 1.0
            _clear_capture(layers)
    count = np.clip(count, 1.0, None)
    mean_peak = sum_peak / count
    mean_valley = sum_valley / count
    del loaded
    torch.cuda.empty_cache()
    return {"peak": mean_peak, "valley": mean_valley}


def plot_grid(results: Dict[str, np.ndarray], labels: List[str], eval_length: int,
              kind: str, output_dir: Path, tag: str = "") -> None:
    n = len(labels)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.4 * nrows), sharex=True)
    axes = np.atleast_1d(axes).reshape(-1)
    mats = [results[lb] for lb in labels]
    vmax = max(np.abs(m).max() for m in mats)
    vmin = -vmax if kind == "valley" else 0.0
    if kind == "peak":
        vmin = min(m.min() for m in mats)
        vmax = max(m.max() for m in mats)
    cmap = "viridis" if kind == "peak" else "viridis_r"
    im = None
    for i, lb in enumerate(labels):
        ax = axes[i]
        im = ax.imshow(results[lb], aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(lb)
        ax.set_xlabel("Length (query position)")
        ax.set_ylabel("layer")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.colorbar(im, ax=axes.tolist(), shrink=0.8, label=f"effective bias {kind}")
    fig.suptitle(f"K1 effective-bias {kind}{(' [' + tag + ']') if tag else ''}: layer x position, L={eval_length}")
    out = output_dir / f"k1_allscales_{kind}_L{eval_length}{('_' + tag) if tag else ''}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--runs_root", type=str, default="runs/pat234_scale_card")
    p.add_argument("--output_dir", type=str, default="pat243_allscales_peak_valley")
    p.add_argument("--eval_lengths", type=int, nargs="+", default=[512, 4096])
    p.add_argument("--num_samples", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="float32")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--micro_batch_size_512", type=int, default=8)
    p.add_argument("--micro_batch_size_4096", type=int, default=2)
    p.add_argument("--restore_router_rms", action="store_true",
                    help="Re-enable the router_logits RMS-norm removed in commit 968a971f81, "
                         "for faithfully re-analyzing checkpoints trained before that commit.")
    p.add_argument("--run_dirs", type=str, nargs="+", default=None,
                    help="Override RUN_DIRS as 'dirname:label' pairs. If omitted, uses the built-in list.")
    p.add_argument("--tag", type=str, default="",
                    help="Suffix appended to output filenames (e.g. 'oldckpt_routerrms').")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.run_dirs:
        run_dirs = [tuple(item.split(":", 1)) for item in args.run_dirs]
    else:
        run_dirs = RUN_DIRS
    tag = f"_{args.tag}" if args.tag else ""

    labels = [lb for _, lb in run_dirs]
    for eval_length in args.eval_lengths:
        bs = args.micro_batch_size_512 if eval_length <= 512 else args.micro_batch_size_4096
        peak_results: Dict[str, np.ndarray] = {}
        valley_results: Dict[str, np.ndarray] = {}
        for dirname, label in run_dirs:
            run_dir = runs_root / dirname
            ckpts = discover_checkpoints(run_dir)
            if 15000 not in ckpts:
                print(f"[skip] {label}: no checkpoint-15000 in {run_dir}")
                continue
            print(f"[main] L={eval_length} label={label} restore_router_rms={args.restore_router_rms} ...")
            out = collect_peak_valley(
                run_dir=run_dir,
                ckpt_dir=ckpts[15000],
                device=args.device,
                dtype_name=args.dtype,
                eval_length=eval_length,
                num_samples=args.num_samples,
                seed=args.seed,
                cache_dir=args.cache_dir,
                micro_batch_size=bs,
                restore_router_rms=args.restore_router_rms,
            )
            peak_results[label] = out["peak"]
            valley_results[label] = out["valley"]
            np.save(output_dir / f"{label}_L{eval_length}_peak{tag}.npy", out["peak"])
            np.save(output_dir / f"{label}_L{eval_length}_valley{tag}.npy", out["valley"])

        present_labels = [lb for lb in labels if lb in peak_results]
        if present_labels:
            plot_grid(peak_results, present_labels, eval_length, "peak", output_dir, tag=tag)
            plot_grid(valley_results, present_labels, eval_length, "valley", output_dir, tag=tag)

    print(f"[main] done. outputs in {output_dir}")


if __name__ == "__main__":
    main()

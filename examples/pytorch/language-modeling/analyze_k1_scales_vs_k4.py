#!/usr/bin/env python
"""Per-scale K4 vs matching single-scale K1 amplitude comparison.

K4 (K4_me8_16_20_24_noC1_s42_sqrtnorm) combines 4 Ricker scales at
me=[8,16,20,24] (rho=[16,256,1024,4096]). This project also has 4
standalone K=1 checkpoints trained at exactly those same me values
(K1_me8_noC1_s42, K1_me16_noC1_s42, K1_me20_noC1_s42, K1_me24_noC1_s42).

This script compares, for each scale index i in K4 (matched to its me
value), two K4-side quantities against the corresponding standalone
K1_meN model's own (K=1) amplitude:
  - raw per-scale basis amplitude (S2_postp99, unweighted -- the pure
    shape/scale amplitude before router gating)
  - router-weighted per-scale contribution amplitude (S3_postgain --
    pi_scale_i * basis_i, i.e. what that scale actually contributes to
    the K4 mixture before summing across scales)

against K1_meN's effective_amp (== mixture_pre == mixture_post for K=1
with multiscale_sum_scale=1.0, all equal, as already confirmed).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import torch

from analyze_k1_k4_wavelet_amp import (
    build_eval_samples,
    causal_centered_rms_amp,
    discover_checkpoints,
    load_model_for_analysis,
    summarize_values,
    _clear_capture,
    _set_capture,
)

ME_TO_K1_DIR = {
    8: "K1_me8_noC1_s42",
    16: "K1_me16_noC1_s42",
    20: "K1_me20_noC1_s42",
    24: "K1_me24_noC1_s42",
}
K4_SCALE_IDX_TO_ME = {0: 8, 1: 16, 2: 20, 3: 24}


def collect_k4_per_scale(layer: Any) -> Dict[int, Dict[str, torch.Tensor]]:
    """Returns {scale_idx: {"raw_amp": [B,Q]-per-q0-chunks concatenated, "weighted_amp": ...}}"""
    cap = getattr(layer, "_pat234_cap", None)
    if cap is None:
        raise RuntimeError("Layer capture dict is not set")
    s2 = cap.get("S2_postp99", [])
    s3 = cap.get("S3_postgain", [])
    if not s2 or not s3:
        raise RuntimeError(f"Missing S2/S3 captures for layer {getattr(layer, 'layer_idx', '?')}")
    raw_by_scale: Dict[int, List[Any]] = defaultdict(list)
    for _lid, scale_idx, q0, basis in s2:
        amp = causal_centered_rms_amp(basis.float(), q0=int(q0))
        raw_by_scale[int(scale_idx)].append(amp)
    weighted_by_scale: Dict[int, List[Any]] = defaultdict(list)
    for _lid, scale_idx, q0, contrib in s3:
        amp = causal_centered_rms_amp(contrib.float(), q0=int(q0))
        weighted_by_scale[int(scale_idx)].append(amp)
    out = {}
    for scale_idx in raw_by_scale:
        out[scale_idx] = {
            "raw_amp": torch.cat(raw_by_scale[scale_idx], dim=1),  # [B, T]
            "weighted_amp": torch.cat(weighted_by_scale[scale_idx], dim=1),
        }
    return out


def collect_k1_amp(layer: Any) -> torch.Tensor:
    cap = getattr(layer, "_pat234_cap", None)
    if cap is None:
        raise RuntimeError("Layer capture dict is not set")
    s4post = cap.get("S4post_postclamp", [])
    if not s4post:
        raise RuntimeError(f"Missing S4post_postclamp for layer {getattr(layer, 'layer_idx', '?')}")
    by_q0 = sorted(s4post, key=lambda e: int(e[1]))
    return torch.cat([causal_centered_rms_amp(chunk.float(), q0=int(q0)) for _lid, q0, chunk in by_q0], dim=1)


def run_model(loaded, samples: torch.Tensor, device: str, batch_size: int, is_k4: bool) -> Dict[str, Any]:
    rows = []
    with torch.no_grad():
        for start in range(0, len(samples), int(batch_size)):
            input_ids = samples[start : start + int(batch_size)].to(device)
            _set_capture(loaded.layers, True)
            loaded.model(input_ids=input_ids, labels=None, use_cache=False)
            for layer in loaded.layers:
                lid = int(getattr(layer, "layer_idx", 0) or 0)
                if is_k4:
                    per_scale = collect_k4_per_scale(layer)
                    for scale_idx, amps in per_scale.items():
                        rows.append(
                            {
                                "layer": lid,
                                "scale_idx": scale_idx,
                                "raw_amp": amps["raw_amp"].numpy(),
                                "weighted_amp": amps["weighted_amp"].numpy(),
                            }
                        )
                else:
                    amp = collect_k1_amp(layer)
                    rows.append({"layer": lid, "amp": amp.numpy()})
            _clear_capture(loaded.layers)
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--k4_run", required=True)
    p.add_argument("--k1_base_dir", default=str(Path(__file__).parent / "runs" / "pat234_scale_card"))
    p.add_argument("--eval_length", type=int, default=512)
    p.add_argument("--num_samples", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["auto", "float32", "float16", "bfloat16"])
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--output_dir", default="analysis_outputs/k1_scales_vs_k4/")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    k4_run = Path(args.k4_run)
    k1_base = Path(args.k1_base_dir)

    k4_ckpts = discover_checkpoints(k4_run)
    k1_ckpts = {}
    for me, dirname in ME_TO_K1_DIR.items():
        k1_ckpts[me] = discover_checkpoints(k1_base / dirname)

    common_steps = sorted(set(k4_ckpts).intersection(*[set(v) for v in k1_ckpts.values()]))
    print(f"Common checkpoint steps across K4 and all 4 K1 references: {common_steps}")

    all_rows = []
    sample_cache = None
    for step in common_steps:
        print(f"=== checkpoint-{step} ===")
        k4_loaded = load_model_for_analysis(k4_ckpts[step], k4_run, args.device, args.dtype)
        if sample_cache is None:
            sample_cache = build_eval_samples(
                k4_loaded.tokenizer, int(args.eval_length), int(args.num_samples), int(args.seed), cache_dir=args.cache_dir
            )
        k4_rows = run_model(k4_loaded, sample_cache, args.device, args.batch_size, is_k4=True)
        del k4_loaded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        k4_by_layer_scale: Dict[tuple, Dict[str, np.ndarray]] = {}
        for r in k4_rows:
            k4_by_layer_scale[(r["layer"], r["scale_idx"])] = {"raw_amp": r["raw_amp"], "weighted_amp": r["weighted_amp"]}

        for me, dirname in ME_TO_K1_DIR.items():
            k1_loaded = load_model_for_analysis(k1_ckpts[me][step], k1_base / dirname, args.device, args.dtype)
            k1_rows = run_model(k1_loaded, sample_cache, args.device, args.batch_size, is_k4=False)
            del k1_loaded
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            k1_by_layer = {r["layer"]: r["amp"] for r in k1_rows}

            scale_idx = [k for k, v in K4_SCALE_IDX_TO_ME.items() if v == me][0]
            for layer in sorted(k1_by_layer):
                k1_amp = k1_by_layer[layer]
                k4_entry = k4_by_layer_scale.get((layer, scale_idx))
                if k4_entry is None:
                    continue
                k1_stats = summarize_values(k1_amp.reshape(-1))
                raw_stats = summarize_values(k4_entry["raw_amp"].reshape(-1))
                weighted_stats = summarize_values(k4_entry["weighted_amp"].reshape(-1))
                all_rows.append(
                    {
                        "checkpoint_step": step,
                        "me": me,
                        "scale_idx": scale_idx,
                        "layer": layer,
                        "k1_amp_mean": k1_stats["mean"],
                        "k4_raw_scale_amp_mean": raw_stats["mean"],
                        "k4_weighted_scale_amp_mean": weighted_stats["mean"],
                        "k4_raw_vs_k1": raw_stats["mean"] / max(k1_stats["mean"], 1e-12),
                        "k4_weighted_vs_k1": weighted_stats["mean"] / max(k1_stats["mean"], 1e-12),
                    }
                )

    df = pd.DataFrame(all_rows)
    df.to_csv(output_dir / "k1_scales_vs_k4_by_layer.csv", index=False)

    summary = (
        df.groupby(["checkpoint_step", "me"])
        .agg(
            k1_amp_mean=("k1_amp_mean", "mean"),
            k4_raw_scale_amp_mean=("k4_raw_scale_amp_mean", "mean"),
            k4_weighted_scale_amp_mean=("k4_weighted_scale_amp_mean", "mean"),
            k4_raw_vs_k1=("k4_raw_vs_k1", "mean"),
            k4_weighted_vs_k1=("k4_weighted_vs_k1", "mean"),
        )
        .reset_index()
        .sort_values(["me", "checkpoint_step"])
    )
    summary.to_csv(output_dir / "k1_scales_vs_k4_summary.csv", index=False)
    print(summary.to_string(index=False))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for me in sorted(ME_TO_K1_DIR):
        sub = summary[summary["me"] == me].sort_values("checkpoint_step")
        axes[0].plot(sub["checkpoint_step"], sub["k4_raw_vs_k1"], marker="o", label=f"me={me}")
        axes[1].plot(sub["checkpoint_step"], sub["k4_weighted_vs_k1"], marker="o", label=f"me={me}")
    for ax, title in zip(axes, ["K4 raw per-scale basis amp / matching K1 amp", "K4 router-weighted per-scale contrib amp / matching K1 amp"]):
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Checkpoint step")
        ax.set_ylabel("ratio")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"K4 per-scale vs matching standalone K1_meN (eval_length={args.eval_length}, n={args.num_samples})")
    fig.tight_layout()
    fig.savefig(output_dir / "k1_scales_vs_k4.png", dpi=150)

    manifest = {
        "k4_run": str(k4_run),
        "k1_dirs": {str(me): str(k1_base / d) for me, d in ME_TO_K1_DIR.items()},
        "common_steps": common_steps,
        "eval_length": args.eval_length,
        "num_samples": args.num_samples,
        "seed": args.seed,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Done. Outputs written under {output_dir}")


if __name__ == "__main__":
    main()

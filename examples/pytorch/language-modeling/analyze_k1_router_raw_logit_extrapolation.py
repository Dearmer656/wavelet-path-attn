"""PAT-243 follow-up: does the ROUTER'S INPUT (q - q_corr, mean over heads) -> raw
pre-sigmoid router_logits[...,0] stay distributionally similar in-range (L<=512,
the training length) vs out-of-range (L>512, extrapolated)?

Motivation (user hypothesis): g0_gate = sigmoid(router_logits[...,0]/tau) was already
shown to stay flat across the L512->L4096 boundary. But sigmoid saturates -- if the
RAW logit keeps drifting while staying in sigmoid's flat tails, g0_gate would look
stable even though the router's input (q - q_corr) is NOT actually stable. This script
checks the raw (pre-sigmoid) logit directly, which is more sensitive to real drift.

Uses the opt-in `_pat_g0_cap["router_logits_raw"]` hook (captures router_logits right
after router_mod(x_feat), before any override/jitter logic). Default off, no effect on
the main training/eval path.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from analyze_k1_k4_wavelet_amp import build_eval_samples, discover_checkpoints, load_model_for_analysis


def collect_raw_logit_and_gate(run_dir: Path, ckpt_dir: Path, device: str, dtype_name: str,
                                eval_length: int, num_samples: int, seed: int, cache_dir: str | None,
                                micro_batch_size: int) -> Dict[str, np.ndarray]:
    loaded = load_model_for_analysis(ckpt_dir, run_dir, device=device, dtype_name=dtype_name)
    layers = loaded.layers
    num_layers = len(layers)

    batches = build_eval_samples(
        loaded.tokenizer, eval_length=eval_length, num_samples=num_samples, seed=seed, cache_dir=cache_dir
    )
    sum_raw = np.zeros((num_layers, eval_length), dtype=np.float64)
    sumsq_raw = np.zeros((num_layers, eval_length), dtype=np.float64)
    sum_gate = np.zeros((num_layers, eval_length), dtype=np.float64)
    count = 0
    bs = int(micro_batch_size)
    with torch.no_grad():
        for start in range(0, batches.shape[0], bs):
            chunk = batches[start:start + bs].to(device)
            for layer in layers:
                layer._pat_g0_cap = {}
            loaded.model(input_ids=chunk)
            per_layer_raw: Dict[int, torch.Tensor] = {}
            per_layer_gate: Dict[int, torch.Tensor] = {}
            for layer in layers:
                cap = layer._pat_g0_cap
                if cap is None or "router_logits_raw" not in cap or "g0_gate" not in cap:
                    raise RuntimeError(
                        f"layer {getattr(layer, 'layer_idx', '?')} missing router_logits_raw/g0_gate in _pat_g0_cap; "
                        "check hook wiring / router_sigmoid_mode."
                    )
                lid, raw = cap["router_logits_raw"][-1]  # [B, T, K+1]
                _, gate = cap["g0_gate"][-1]              # [B, T, 1]
                per_layer_raw[int(lid)] = raw[..., 0]      # null-column raw logit, [B, T]
                per_layer_gate[int(lid)] = gate.squeeze(-1)
            for lid in range(num_layers):
                raw0 = per_layer_raw[lid]
                sum_raw[lid] += raw0.sum(dim=0).cpu().numpy()
                sumsq_raw[lid] += raw0.pow(2).sum(dim=0).cpu().numpy()
                sum_gate[lid] += per_layer_gate[lid].sum(dim=0).cpu().numpy()
            count += chunk.shape[0]
            for layer in layers:
                layer._pat_g0_cap = None
    mean_raw = sum_raw / max(1, count)
    var_raw = sumsq_raw / max(1, count) - mean_raw ** 2
    std_raw = np.sqrt(np.clip(var_raw, 0.0, None))
    mean_gate = sum_gate / max(1, count)
    del loaded
    torch.cuda.empty_cache()
    return {"mean_raw_logit": mean_raw, "std_raw_logit": std_raw, "mean_gate": mean_gate}


def plot_comparison(result: Dict[str, np.ndarray], train_length: int, output_dir: Path) -> None:
    mean_raw = result["mean_raw_logit"]
    std_raw = result["std_raw_logit"]
    mean_gate = result["mean_gate"]
    T = mean_raw.shape[1]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for ax, mat, title, cmap in [
        (axes[0], mean_raw, "raw pre-sigmoid logit: mean(layer, position)", "PuOr"),
        (axes[1], std_raw, "raw pre-sigmoid logit: std-across-batch(layer, position)", "viridis"),
        (axes[2], mean_gate, "post-sigmoid g0_gate: mean(layer, position) (reference)", "viridis"),
    ]:
        vmax = np.abs(mat).max() if "PuOr" in cmap else mat.max()
        vmin = -vmax if "PuOr" in cmap else mat.min()
        im = ax.imshow(mat, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.axvline(train_length - 0.5, color="red", linestyle="--", linewidth=1, label=f"train length={train_length}")
        ax.set_title(title)
        ax.set_ylabel("layer")
        fig.colorbar(im, ax=ax, shrink=0.85)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Length (query position)")
    fig.suptitle("K1_me16 center0: raw router_logits[...,0] vs post-sigmoid g0_gate, L=%d" % T)
    out = output_dir / "k1_me16_raw_logit_vs_gate_extrapolation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")

    # distance-binned summary, mirroring the earlier g0_gate distance-bin check
    bins = [(0, train_length), (train_length, 2 * train_length), (2 * train_length, 4 * train_length),
            (4 * train_length, T)] if T > train_length else [(0, T)]
    bins = [(lo, hi) for lo, hi in bins if hi > lo and lo < T]
    print("=== distance-binned raw logit mean (avg over layers) ===")
    for lo, hi in bins:
        hi = min(hi, T)
        seg = mean_raw[:, lo:hi]
        print(f"pos [{lo:5d},{hi:5d}): mean={seg.mean():+.4f}  std_within_bin={seg.std():.4f}")
    print("=== distance-binned g0_gate mean (avg over layers), reference ===")
    for lo, hi in bins:
        hi = min(hi, T)
        seg = mean_gate[:, lo:hi]
        print(f"pos [{lo:5d},{hi:5d}): mean={seg.mean():.4f}  std_within_bin={seg.std():.4f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str,
                    default="runs/pat234_scale_card/K1_me16_noC1_s42_center0_ricker_norouterrms_lang01_a100x2")
    p.add_argument("--checkpoint_step", type=int, default=15000)
    p.add_argument("--output_dir", type=str, default="pat243_g0_gate_analysis_L4096")
    p.add_argument("--eval_length", type=int, default=4096)
    p.add_argument("--train_length", type=int, default=512)
    p.add_argument("--num_samples", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="float32")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--micro_batch_size", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpts = discover_checkpoints(run_dir)
    if args.checkpoint_step not in ckpts:
        raise RuntimeError(f"checkpoint-{args.checkpoint_step} not found in {run_dir}. Found: {sorted(ckpts)}")

    result = collect_raw_logit_and_gate(
        run_dir=run_dir,
        ckpt_dir=ckpts[args.checkpoint_step],
        device=args.device,
        dtype_name=args.dtype,
        eval_length=args.eval_length,
        num_samples=args.num_samples,
        seed=args.seed,
        cache_dir=args.cache_dir,
        micro_batch_size=args.micro_batch_size,
    )
    for name, mat in result.items():
        np.save(output_dir / f"{name}_ckpt{args.checkpoint_step}.npy", mat)
    plot_comparison(result, train_length=args.train_length, output_dir=output_dir)
    print(f"[main] done. outputs in {output_dir}")


if __name__ == "__main__":
    main()

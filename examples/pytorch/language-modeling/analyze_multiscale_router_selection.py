#!/usr/bin/env python
"""Diagnostic: for K3/K5 (multi-scale, independent_rms), does the router
concentrate onto a single scale (matching K1's best point) or stay diffuse
across candidates? Distinguishes two competing explanations for why K3/K5
underperform K1's best single-scale run despite being centered on it.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from analyze_k1_k4_wavelet_amp import build_eval_samples, load_model_for_analysis


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_root", default="runs/pat234_scale_card")
    p.add_argument("--run_dirs", nargs="+", required=True, help="dirname:label pairs")
    p.add_argument("--eval_length", type=int, default=512)
    p.add_argument("--num_samples", type=int, default=32)
    p.add_argument("--micro_batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float32")
    return p.parse_args()


def main():
    args = parse_args()
    for spec in args.run_dirs:
        dirname, label = spec.split(":", 1)
        run_dir = Path(args.runs_root) / dirname
        ckpt = run_dir / "checkpoint-15000"
        loaded = load_model_for_analysis(ckpt, run_dir, args.device, args.dtype)
        model, tokenizer, layers, manifest = (
            loaded.model,
            loaded.tokenizer,
            loaded.layers,
            loaded.manifest,
        )
        K = int(manifest["wavelet_ctxscale_k"])
        scales = manifest["scale_list"]
        print(f"\n=== {label} ({dirname}) K={K} scales(rho)={[round(s, 1) for s in scales]} ===", flush=True)

        samples = build_eval_samples(tokenizer, args.eval_length, args.num_samples, args.seed)
        bs = args.micro_batch_size
        n_layers = len(layers)
        sum_prob = [torch.zeros(K, dtype=torch.float64) for _ in range(n_layers)]
        sum_null = [0.0 for _ in range(n_layers)]
        count = [0 for _ in range(n_layers)]

        with torch.no_grad():
            for i in range(0, len(samples), bs):
                batch = samples[i : i + bs].to(args.device)
                model(input_ids=batch)
                for li, layer in enumerate(layers):
                    prob = getattr(layer, "_last_ctxscale_router_prob", None)
                    null_mass = getattr(layer, "_last_ctxscale_null_mass", None)
                    if prob is None:
                        continue
                    p = prob.mean(dim=(0, 1, 2)).double().cpu()
                    sum_prob[li] += p
                    count[li] += 1
                    if null_mass is not None:
                        sum_null[li] += float(null_mass.mean().item())

        layer_avgs = []
        for li in range(n_layers):
            if count[li] == 0:
                print(f"  layer {li:2d}: no router_prob captured")
                continue
            avg = (sum_prob[li] / count[li]).numpy()
            avg_null = sum_null[li] / count[li]
            avg_norm = avg / max(avg.sum(), 1e-12)
            entropy = -np.sum(avg_norm * np.log(avg_norm + 1e-12)) / np.log(K)
            top_idx = int(np.argmax(avg))
            layer_avgs.append(avg)
            print(
                f"  layer {li:2d}: pi_scale={np.round(avg, 4).tolist()} null_mass={avg_null:.4f} "
                f"norm_entropy={entropy:.3f} top_scale=rho{scales[top_idx]:.0f}",
                flush=True,
            )

        if layer_avgs:
            overall = np.mean(np.stack(layer_avgs, axis=0), axis=0)
            overall_norm = overall / max(overall.sum(), 1e-12)
            overall_entropy = -np.sum(overall_norm * np.log(overall_norm + 1e-12)) / np.log(K)
            top_idx = int(np.argmax(overall))
            print(
                f"  OVERALL (mean over layers): pi_scale={np.round(overall, 4).tolist()} "
                f"norm_entropy={overall_entropy:.3f} top_scale=rho{scales[top_idx]:.0f}",
                flush=True,
            )

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

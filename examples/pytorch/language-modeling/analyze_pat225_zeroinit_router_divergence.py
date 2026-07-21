#!/usr/bin/env python3
"""
PAT-225 zero-init mechanism test: does zero-init make the router CONVERGE to a
more seed-consistent state than kaiming?

kaiming routers start from independent random inits -> earlier analysis found
near-orthogonal routers across seeds (cos ~0.02-0.05). zero-init routers all
start from the SAME point (all zeros), so any seed-to-seed divergence at
ckpt-15000 is purely training-dynamics driven. If zero-init s42-vs-s43 routers
end up MORE aligned than kaiming s42-vs-s43, that is the mechanistic explanation
for zero-init's variance reduction.

Compares, at ckpt-15000, K4:
  cos(kaiming_s42, kaiming_s43)  vs  cos(zeroinit_s42, zeroinit_s43)
per layer for the router weight/bias, plus router weight norms (does zero-init
keep the router smaller?). CPU-only static weight comparison, no GPU.
"""
import numpy as np
from safetensors import safe_open
import torch

BASE = "/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat225_scale_card"
STEP = 15000
N_LAYERS = 12
ROUTER_KEYS = ["attn.core.wavelet_ctx_router.weight", "attn.core.wavelet_ctx_router.bias"]

RUNS = {
    "kaiming_s42": f"{BASE}/S4_s42/checkpoint-{STEP}/model.safetensors",
    "kaiming_s43": f"{BASE}/S4_s43/checkpoint-{STEP}/model.safetensors",
    "zeroinit_s42": f"{BASE}/S4_s42_zeroinit/checkpoint-{STEP}/model.safetensors",
    "zeroinit_s43": f"{BASE}/S4_s43_zeroinit/checkpoint-{STEP}/model.safetensors",
}


def load_router(path, layer):
    prefix = f"transformer.h.{layer}."
    out = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            if k.startswith(prefix):
                short = k[len(prefix):]
                if short in ROUTER_KEYS:
                    out[short] = f.get_tensor(k)
    return out


def cat_router(path, layer):
    d = load_router(path, layer)
    return torch.cat([d[k].flatten().float() for k in ROUTER_KEYS])


def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def main():
    print(f"=== Router weight-space: kaiming vs zero-init seed divergence @ ckpt-{STEP} (K4) ===\n")
    print(f"{'L':>3} | {'cos(kai42,kai43)':>17} | {'cos(zero42,zero43)':>19} | "
          f"{'||kai42||':>10} {'||kai43||':>10} | {'||zero42||':>11} {'||zero43||':>11}")
    print("-" * 100)
    kai_cos, zero_cos = [], []
    kai_norm, zero_norm = [], []
    for L in range(N_LAYERS):
        k42 = cat_router(RUNS["kaiming_s42"], L)
        k43 = cat_router(RUNS["kaiming_s43"], L)
        z42 = cat_router(RUNS["zeroinit_s42"], L)
        z43 = cat_router(RUNS["zeroinit_s43"], L)
        ck = cos(k42, k43)
        cz = cos(z42, z43)
        kai_cos.append(ck); zero_cos.append(cz)
        nk42, nk43 = float(k42.norm()), float(k43.norm())
        nz42, nz43 = float(z42.norm()), float(z43.norm())
        kai_norm += [nk42, nk43]; zero_norm += [nz42, nz43]
        print(f"{L:>3} | {ck:>17.4f} | {cz:>19.4f} | {nk42:>10.3f} {nk43:>10.3f} | {nz42:>11.3f} {nz43:>11.3f}")
    print("-" * 100)
    print(f"MEAN cos: kaiming(42,43)={np.mean(kai_cos):.4f}  zeroinit(42,43)={np.mean(zero_cos):.4f}")
    print(f"MEAN router weight norm: kaiming={np.mean(kai_norm):.3f}  zeroinit={np.mean(zero_norm):.3f}")
    print()
    if np.mean(zero_cos) > np.mean(kai_cos) + 0.05:
        print(">> zero-init routers are MORE seed-aligned -> supports variance-reduction mechanism")
    elif abs(np.mean(zero_cos) - np.mean(kai_cos)) <= 0.05:
        print(">> zero-init routers are NOT meaningfully more aligned than kaiming "
              "-> seed divergence is training-dynamics driven, not init-driven")
    else:
        print(">> zero-init routers are LESS aligned than kaiming (unexpected)")


if __name__ == "__main__":
    main()

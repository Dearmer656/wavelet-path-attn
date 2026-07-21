#!/usr/bin/env python3
"""Verify zero-init actually changed the router trajectory: same seed, only
init differs. If cos(kaiming_sX, zeroinit_sX) router ~ 1.0 the flag was a
no-op (BUG); if clearly < 1 zero-init took effect."""
from safetensors import safe_open
import torch

BASE = "/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat225_scale_card"
ROUTER = ["attn.core.wavelet_ctx_router.weight", "attn.core.wavelet_ctx_router.bias"]
N_LAYERS = 12


def cat_router(path, layer):
    prefix = f"transformer.h.{layer}."
    d = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            s = k[len(prefix):] if k.startswith(prefix) else None
            if s in ROUTER:
                d[s] = f.get_tensor(k)
    return torch.cat([d[k].flatten().float() for k in ROUTER])


def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


for step in (2500, 15000):
    print(f"\n=== ckpt-{step}: same-seed kaiming vs zeroinit router cos (per layer) ===")
    for seed in ("s42", "s43"):
        kp = f"{BASE}/S4_{seed}/checkpoint-{step}/model.safetensors"
        zp = f"{BASE}/S4_{seed}_zeroinit/checkpoint-{step}/model.safetensors"
        import os
        if not (os.path.exists(kp) and os.path.exists(zp)):
            print(f"  {seed}: missing ckpt (k={os.path.exists(kp)} z={os.path.exists(zp)})")
            continue
        cs = [cos(cat_router(kp, L), cat_router(zp, L)) for L in range(N_LAYERS)]
        mean = sum(cs) / len(cs)
        verdict = "NO-OP (identical!)" if mean > 0.99 else ("took effect" if mean < 0.9 else "PARTIAL/unclear")
        print(f"  {seed}: mean cos(kaiming,zeroinit)={mean:.4f}  per-layer={[round(c,3) for c in cs]}  -> {verdict}")

#!/usr/bin/env python3
"""
PAT-225 K4 outlier investigation: compare actual WEIGHT tensors (not just
downstream attention activations) between s42/s43/s44 at checkpoint-5000
(earliest available for all three) to see which parameter groups already
diverge -- router only, or backbone (Q/K/V/O, PaTH's low-rank W generator,
beta generator) too. This is a static weight-space comparison, no forward
pass / no GPU needed.
"""
import json
from pathlib import Path
from safetensors import safe_open
import torch

BASE = "/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat225_scale_card"
SEEDS = ("s42", "s43", "s44")
STEP = 5000
N_LAYERS = 12

GROUPS = {
    "router": ["attn.core.wavelet_ctx_router.weight", "attn.core.wavelet_ctx_router.bias"],
    "q_proj": ["attn.core.q_proj.weight"],
    "k_proj": ["attn.core.k_proj.weight"],
    "v_proj": ["attn.core.v_proj.weight"],
    "o_proj": ["attn.core.o_proj.weight"],
    "w_proj (path low-rank W gen)": ["attn.core.w_proj.0.weight", "attn.core.w_proj.1.weight"],
    "bt_proj (beta gen)": ["attn.core.bt_proj.weight", "attn.core.bt_proj.bias"],
}


def load_layer_tensors(seed, layer):
    ckpt = f"{BASE}/S4_{seed}/checkpoint-{STEP}/model.safetensors"
    prefix = f"transformer.h.{layer}."
    out = {}
    with safe_open(ckpt, framework="pt") as f:
        for k in f.keys():
            if k.startswith(prefix):
                out[k[len(prefix):]] = f.get_tensor(k)
    return out


def cos_sim(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def rel_l2(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return float((a - b).norm() / a.norm().clamp_min(1e-8))


def main():
    all_layer_tensors = {seed: {} for seed in SEEDS}
    for seed in SEEDS:
        for L in range(N_LAYERS):
            all_layer_tensors[seed][L] = load_layer_tensors(seed, L)

    print(f"=== Weight-space divergence at checkpoint-{STEP} (cosine sim, all 3 pairs) ===")
    header = f"{'L':>3} {'group':>28} {'cos(42,43)':>11} {'cos(42,44)':>11} {'cos(43,44)':>11}"
    print(header)
    agg = {g: {"cos4243": [], "cos4244": [], "cos4344": []} for g in GROUPS}
    for L in range(N_LAYERS):
        for gname, keys in GROUPS.items():
            try:
                t42 = torch.cat([all_layer_tensors["s42"][L][k].flatten() for k in keys])
                t43 = torch.cat([all_layer_tensors["s43"][L][k].flatten() for k in keys])
                t44 = torch.cat([all_layer_tensors["s44"][L][k].flatten() for k in keys])
            except KeyError as e:
                print(f"  L{L} {gname}: missing key {e}")
                continue
            c4243, c4244, c4344 = cos_sim(t42, t43), cos_sim(t42, t44), cos_sim(t43, t44)
            agg[gname]["cos4243"].append(c4243)
            agg[gname]["cos4244"].append(c4244)
            agg[gname]["cos4344"].append(c4344)
            print(f"{L:>3} {gname:>28} {c4243:>11.4f} {c4244:>11.4f} {c4344:>11.4f}")

    print("\n=== Mean cosine similarity by parameter group, averaged over all 12 layers ===")
    for gname, vals in agg.items():
        if vals["cos4243"]:
            m4243 = sum(vals["cos4243"]) / len(vals["cos4243"])
            m4244 = sum(vals["cos4244"]) / len(vals["cos4244"])
            m4344 = sum(vals["cos4344"]) / len(vals["cos4344"])
            print(f"  {gname:>28}: cos(42,43)={m4243:.4f}  cos(42,44)={m4244:.4f}  cos(43,44)={m4344:.4f}")

    out_path = Path(__file__).parent / "results" / "pat225_K4_weight_divergence.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({g: v for g, v in agg.items()}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

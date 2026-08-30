#!/usr/bin/env python3
"""Placement-stratified F1 at L4096 for rho128/rho256 seed42 vs PA-only seed42,
using genuinely fresh (non-cached) inference for rho128/rho256 and PA-only.
"""
import json, math
from pathlib import Path
import numpy as np

RESULTS = Path("/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/results_uniform")
UNIFORM_JSONL = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/hotpot_long_dev_uniform.jsonl"
LENGTH = 4096

CONFIGS = {
    "rho128_s42_fresh": "K1_L512_me14_rho128_ricker_s42_ckpt15000_new2",
    "rho256_s42_fresh": "K1_L512_me16_rho256_ricker_rerun_s42_ckpt15000_new2",
    "PA_only_s42_fresh": "PA_only_s42_uniform_new2",
    "PA_only_s42_OLD_cached": "PA_only_s42_uniform",
}

DISTRIBUTIONS = {
    "front_33":   lambda pct: pct < 0.33,
    "middle_40":  lambda pct: 0.30 <= pct <= 0.70,
    "back_33":    lambda pct: pct > 0.67,
    "bimodal_20": lambda pct: pct < 0.2 or pct > 0.8,
    "uniform":    lambda pct: True,
}


def bootstrap_mean(values, B=2000, seed=42):
    arr = np.array(values)
    n = len(arr)
    if n < 2:
        return (float(arr.mean()) if n == 1 else float("nan")), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.array([arr[rng.integers(0, n, n)].mean() for _ in range(B)])
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load_uniform_records():
    out = []
    with open(UNIFORM_JSONL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            meta = d.get("meta", {})
            out.append(float(meta.get("placement_actual_pct", float("nan"))))
    return out


def load_f1_by_index(dname):
    rec_path = RESULTS / dname / f"L{LENGTH}" / f"L{LENGTH}_1.0scale_eval_records_baseline.jsonl"
    out = {}
    if not rec_path.exists():
        return out
    with open(rec_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            idx = r.get("sample_index")
            f1 = r.get("f1")
            if idx is not None and f1 is not None:
                out[idx] = float(f1)
    return out


placement_pct = load_uniform_records()
print(f"Loaded {len(placement_pct)} uniform placement records\n")

f1_by_cfg = {name: load_f1_by_index(dname) for name, dname in CONFIGS.items()}
for name, f1s in f1_by_cfg.items():
    print(f"{name}: n={len(f1s)}")
print()

pa_key = "PA_only_s42_fresh" if len(f1_by_cfg["PA_only_s42_fresh"]) > 0 else "PA_only_s42_OLD_cached"
pa_label = "PA-only (FRESH)" if pa_key == "PA_only_s42_fresh" else "PA-only (OLD cached, placeholder until fresh lands)"
print(f"=== Using PA-only baseline: {pa_label} ===\n")
pa_f1 = f1_by_cfg[pa_key]

for rho_name in ["rho128_s42_fresh", "rho256_s42_fresh"]:
    wr_f1 = f1_by_cfg[rho_name]
    print(f"{'='*70}\n{rho_name} vs {pa_label} (L4096)\n{'='*70}")
    print(f"{'Bucket':<12} {'n':>5} {'PA F1':>8} {'QWAB F1':>8} {'Delta':>8} {'95% CI (Delta)':>20}")
    for dist_name, dist_fn in DISTRIBUTIONS.items():
        idxs = [
            idx for idx in range(len(placement_pct))
            if not math.isnan(placement_pct[idx]) and dist_fn(placement_pct[idx])
            and idx in pa_f1 and idx in wr_f1
        ]
        if not idxs:
            continue
        pa_vals = np.array([pa_f1[i] for i in idxs])
        wr_vals = np.array([wr_f1[i] for i in idxs])
        diffs = wr_vals - pa_vals
        d_mean, d_lo, d_hi = bootstrap_mean(diffs)
        print(f"{dist_name:<12} {len(idxs):>5} {pa_vals.mean():>8.4f} {wr_vals.mean():>8.4f} "
              f"{d_mean:>+8.4f} [{d_lo:+.4f}, {d_hi:+.4f}]")
    print()

#!/usr/bin/env python3
"""Placement-stratified F1 for rho128/rho256 vs PA-only at L512/L2048,
reusing existing per-example uniform-eval records (no new inference).
Explicit checkpoint dirs (not substring-matched) to avoid picking up the
many unrelated ablation/swap variants that litter results_uniform/.
"""
import json, math
from pathlib import Path
import numpy as np

RESULTS = Path("/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/results_uniform")
UNIFORM_JSONL = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/hotpot_long_dev_uniform.jsonl"
LENGTHS = [512, 2048]

PA_DIRS = ["PA_only_s42_uniform", "PA_only_s43_uniform", "PA_only_s44_uniform"]
RHO_DIRS = {
    "rho128": [
        "K1_L512_me14_rho128_ricker_s42_ckpt15000",
        "K1_L512_me14_rho128_ricker_s43_ckpt15000",
        "K1_L512_me14_rho128_ricker_s44_ckpt15000",
    ],
    "rho256": [
        "K1_L512_me16_rho256_ricker_rerun_s42_ckpt15000",
        "K1_L512_me16_rho256_seed43_ckpt15000",
        "K1_L512_me16_rho256_seed44_ckpt15000",
    ],
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
        m = float(arr.mean()) if n == 1 else float("nan")
        return m, float("nan"), float("nan")
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


def load_f1_by_index(rec_path):
    out = {}
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


def collect_avg_f1(dir_names, length):
    """Average per-sample-index F1 across the given seed dirs."""
    per_idx = {}
    for dname in dir_names:
        rec_path = RESULTS / dname / f"L{length}" / f"L{length}_1.0scale_eval_records_baseline.jsonl"
        f1s = load_f1_by_index(rec_path)
        for idx, f1 in f1s.items():
            per_idx.setdefault(idx, []).append(f1)
    return {idx: float(np.mean(v)) for idx, v in per_idx.items()}


placement_pct = load_uniform_records()
print(f"Loaded {len(placement_pct)} uniform placement records")

for rho_name, rho_dirs in RHO_DIRS.items():
    print(f"\n{'='*70}\n{rho_name} vs PA-only\n{'='*70}")
    for length in LENGTHS:
        pa_f1 = collect_avg_f1(PA_DIRS, length)
        wr_f1 = collect_avg_f1(rho_dirs, length)
        print(f"\n--- L{length} ---")
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

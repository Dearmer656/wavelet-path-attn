#!/usr/bin/env python3
"""Aggregate the offset-shift stress test: PA-only vs sig_delta_detach (QWAB),
paired per-example F1, 3 seeds x 4 shifts, bootstrap 95% CI on the mean delta.
Uses full-size (non-_n500) eval record jsonls.
"""
import json
import statistics as st
from pathlib import Path
import numpy as np

BASE = Path("/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/results_offset_shift")
SHIFTS = [0, 256, 512, 1024]
SEEDS = [42, 43, 44]

def load_f1(path):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r["example_id"]] = r["f1"]
    return out

def bootstrap_ci(diffs, B=2000, seed=0):
    arr = np.array(diffs)
    n = len(arr)
    rng = np.random.default_rng(seed)
    means = np.array([arr[rng.integers(0, n, n)].mean() for _ in range(B)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(arr.mean()), float(lo), float(hi)

results = {}
for k in SHIFTS:
    per_seed = []
    for s in SEEDS:
        pa_path = BASE / f"PA_only_s{s}" / f"shift{k}" / "L4096_1.0scale_eval_records_baseline.jsonl"
        qwab_path = BASE / f"sig_delta_detach_s{s}_delta_detach" / f"shift{k}" / "L4096_1.0scale_eval_records_baseline.jsonl"
        pa_f1 = load_f1(pa_path)
        qwab_f1 = load_f1(qwab_path)
        common = sorted(set(pa_f1) & set(qwab_f1))
        pa_vals = np.array([pa_f1[e] for e in common])
        qwab_vals = np.array([qwab_f1[e] for e in common])
        diffs = qwab_vals - pa_vals
        per_seed.append({
            "seed": s, "n": len(common),
            "pa_mean": float(pa_vals.mean()), "qwab_mean": float(qwab_vals.mean()),
            "delta_mean": float(diffs.mean()),
        })
    results[k] = per_seed

print(f"{'shift':>6} {'seed':>5} {'n':>6} {'PA F1':>8} {'QWAB F1':>8} {'Delta':>8}")
for k in SHIFTS:
    for row in results[k]:
        print(f"{k:>6} {row['seed']:>5} {row['n']:>6} {row['pa_mean']:>8.4f} {row['qwab_mean']:>8.4f} {row['delta_mean']:>+8.4f}")

print()
print("=== 3-seed mean +- std, per shift ===")
print(f"{'shift':>6} {'PA mean+-std':>16} {'QWAB mean+-std':>16} {'Delta mean+-std':>18}")
for k in SHIFTS:
    pa_vals = [r["pa_mean"] for r in results[k]]
    qwab_vals = [r["qwab_mean"] for r in results[k]]
    delta_vals = [r["delta_mean"] for r in results[k]]
    pa_m, pa_s = st.mean(pa_vals), st.pstdev(pa_vals)
    qwab_m, qwab_s = st.mean(qwab_vals), st.pstdev(qwab_vals)
    d_m, d_s = st.mean(delta_vals), st.pstdev(delta_vals)
    print(f"{k:>6} {pa_m:.4f}+-{pa_s:.4f}   {qwab_m:.4f}+-{qwab_s:.4f}   {d_m:+.4f}+-{d_s:.4f}")

print()
print("=== Pooled paired bootstrap CI on Delta (all seeds' matched examples pooled per shift) ===")
for k in SHIFTS:
    all_diffs = []
    for s in SEEDS:
        pa_path = BASE / f"PA_only_s{s}" / f"shift{k}" / "L4096_1.0scale_eval_records_baseline.jsonl"
        qwab_path = BASE / f"sig_delta_detach_s{s}_delta_detach" / f"shift{k}" / "L4096_1.0scale_eval_records_baseline.jsonl"
        pa_f1 = load_f1(pa_path)
        qwab_f1 = load_f1(qwab_path)
        common = sorted(set(pa_f1) & set(qwab_f1))
        all_diffs.extend([qwab_f1[e] - pa_f1[e] for e in common])
    mean, lo, hi = bootstrap_ci(all_diffs)
    print(f"shift={k}: n_pooled={len(all_diffs)}  Delta={mean:+.4f}  95% CI=[{lo:+.4f}, {hi:+.4f}]")

out = {
    "per_seed": {str(k): results[k] for k in SHIFTS},
}
with open("/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/results_offset_shift/summary_pa_vs_qwab.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nsaved summary_pa_vs_qwab.json")

print()
print("=== Pooled PA/QWAB F1 means (for table) ===")
for k in SHIFTS:
    pa_all, qwab_all = [], []
    for s in SEEDS:
        pa_path = BASE / f"PA_only_s{s}" / f"shift{k}" / "L4096_1.0scale_eval_records_baseline.jsonl"
        qwab_path = BASE / f"sig_delta_detach_s{s}_delta_detach" / f"shift{k}" / "L4096_1.0scale_eval_records_baseline.jsonl"
        pa_f1 = load_f1(pa_path)
        qwab_f1 = load_f1(qwab_path)
        common = sorted(set(pa_f1) & set(qwab_f1))
        pa_all.extend([pa_f1[e] for e in common])
        qwab_all.extend([qwab_f1[e] for e in common])
    print(f"shift={k}: n={len(pa_all)}  PA={np.mean(pa_all):.4f}  QWAB={np.mean(qwab_all):.4f}")

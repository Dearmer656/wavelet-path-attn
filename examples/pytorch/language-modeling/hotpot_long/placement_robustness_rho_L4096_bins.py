#!/usr/bin/env python3
"""Placement-stratified F1 at L4096 for rho128/rho256 seed42 vs PA-only seed42,
using fixed 20%-wide location bins (0-20%, 20-40%, ..., 80-100%) instead of
front/middle/back/bimodal buckets.
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

BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]


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
f1_by_cfg = {name: load_f1_by_index(dname) for name, dname in CONFIGS.items()}

pa_key = "PA_only_s42_fresh" if len(f1_by_cfg["PA_only_s42_fresh"]) > 0 else "PA_only_s42_OLD_cached"
pa_label = "PA-only (FRESH)" if pa_key == "PA_only_s42_fresh" else "PA-only (OLD cached, placeholder)"
pa_f1 = f1_by_cfg[pa_key]
print(f"=== PA-only baseline in use: {pa_label} ===\n")

results = {}
for rho_name in ["rho128_s42_fresh", "rho256_s42_fresh"]:
    wr_f1 = f1_by_cfg[rho_name]
    rows = []
    for lo, hi in BINS:
        idxs = [
            idx for idx in range(len(placement_pct))
            if not math.isnan(placement_pct[idx]) and lo <= placement_pct[idx] < hi
            and idx in pa_f1 and idx in wr_f1
        ]
        pa_vals = np.array([pa_f1[i] for i in idxs])
        wr_vals = np.array([wr_f1[i] for i in idxs])
        diffs = wr_vals - pa_vals
        d_mean, d_lo, d_hi = bootstrap_mean(diffs)
        rows.append((lo, hi, len(idxs), pa_vals.mean(), wr_vals.mean(), d_mean, d_lo, d_hi))
    results[rho_name] = rows
    print(f"{rho_name} vs {pa_label}")
    print(f"{'Bin':<10} {'n':>5} {'PA F1':>8} {'QWAB F1':>8} {'Delta':>8} {'95% CI':>20}")
    for lo, hi, n, pa_m, wr_m, d_m, d_lo, d_hi in rows:
        print(f"{int(lo*100)}-{int(hi*100)}%   {n:>5} {pa_m:>8.4f} {wr_m:>8.4f} {d_m:>+8.4f} [{d_lo:+.4f}, {d_hi:+.4f}]")
    print()

# --- LaTeX table ---
print("=== LaTeX ===\n")
lines = []
lines.append(r"\begin{table}[t]")
lines.append(r"\centering")
lines.append(r"\begin{tabular}{lccc}")
lines.append(r"\toprule")
lines.append(r"Evidence position & PA-only F1 & QWAB F1 & $\Delta$F1 (95\% CI) \\")
lines.append(r"\midrule")
for rho_name, tag in [("rho128_s42_fresh", r"$\rho=128$"), ("rho256_s42_fresh", r"$\rho=256$")]:
    rows = results[rho_name]
    lines.append(r"\multicolumn{4}{l}{\textit{" + tag + r"}} \\")
    for lo, hi, n, pa_m, wr_m, d_m, d_lo, d_hi in rows:
        bin_label = f"{int(lo*100)}\\%--{int(hi*100)}\\%"
        lines.append(
            f"{bin_label} & {pa_m:.4f} & {wr_m:.4f} & "
            f"{d_m:+.4f} [{d_lo:+.4f}, {d_hi:+.4f}] \\\\"
        )
    if rho_name == "rho128_s42_fresh":
        lines.append(r"\addlinespace")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\caption{L4096 F1 by evidence position bin (seed 42).}")
lines.append(r"\label{tab:placement-l4096}")
lines.append(r"\end{table}")
latex = "\n".join(lines)
print(latex)

with open("/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/paper_results/scale_sweep_charts/placement_l4096_table.tex", "w") as f:
    f.write(latex + "\n")

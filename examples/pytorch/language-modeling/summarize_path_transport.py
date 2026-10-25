#!/usr/bin/env python3
"""Aggregate probe_path_transport_far_near.py outputs: family x length, mean±std over seeds (ddof=0)."""
import json, sys
from pathlib import Path
import numpy as np

D = Path(sys.argv[1] if len(sys.argv) > 1 else
         "/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat310_c2/transport_probe")
FAMS = ["PaTH", "QWAB", "box055", "boxsb086"]
SEEDS = ["s42", "s43", "s44"]
LENS = [512, 2048, 4096]


def load(f, L):
    out = []
    for s in SEEDS:
        p = D / f"{f}_{s}_L{L}.json"
        if p.exists(): out.append(json.loads(p.read_text()))
    return out


def ms(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return f"{np.mean(v):.3f}±{np.std(v):.3f}" if v else "-"


def layer_avg(r, get, layers=None):
    ls = layers or sorted(r["layers"], key=int)
    vals = [get(r["layers"][l]) for l in ls]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def safe(fn):
    def g(x):
        try: return fn(x)
        except (KeyError, TypeError, IndexError): return None
    return g


print("## A. distance bins (layer-avg): corr(C,E) = content fidelity through PaTH transport; R/C = transport/content rms")
for L in LENS:
    print(f"\n### L{L}")
    bins = None
    rows = {}
    for f in FAMS:
        rs = load(f, L)
        if not rs: continue
        bins = rs[0]["bins"]
        nb = len(bins) - 1
        rows[f] = [(ms([layer_avg(r, safe(lambda x, b=b: x["bin"][b]["corrCE"])) for r in rs]),
                    ms([layer_avg(r, safe(lambda x, b=b: x["bin"][b]["rmsR"] / x["bin"][b]["rmsC"])) for r in rs]),
                    ms([layer_avg(r, safe(lambda x, b=b: x["bin"][b]["massF"])) for r in rs]))
                   for b in range(nb)]
    if not rows: continue
    hdr = [f"{bins[b]}-{bins[b+1]}" for b in range(len(bins) - 1) if bins[b] < L]
    print("| model | metric | " + " | ".join(hdr) + " |")
    print("|---|---|" + "---|" * len(hdr))
    for f, cells in rows.items():
        for k, name in enumerate(["corrCE", "R/C", "massF"]):
            print(f"| {f} | {name} | " + " | ".join(c[k] for c in cells[:len(hdr)]) + " |")

print("\n## B. absolute regions (layer-avg)")
KEYS = ["massE", "massF", "corrCE", "rmsB", "meanB", "corrBR", "corrBE", "corrBC"]
for L in LENS:
    print(f"\n### L{L}")
    print("| model | region | " + " | ".join(KEYS) + " |")
    print("|---|---|" + "---|" * len(KEYS))
    for f in FAMS:
        rs = load(f, L)
        if not rs: continue
        for rn in ("prefix", "far_mid", "near"):
            print(f"| {f} | {rn} | " + " | ".join(
                ms([layer_avg(r, safe(lambda x, k=k: x["region"][rn][k])) for r in rs]) for k in KEYS) + " |")

print("\n## C. support docs vs distance-matched distractors (layer-avg; snr = mean over heads)")
SK = ["dist_mean", "snrC", "snrNegR", "snrE", "snrF", "massE", "massF"]
for L in LENS:
    print(f"\n### L{L}")
    print("| model | " + " | ".join(SK) + " |")
    print("|---|" + "---|" * len(SK))
    for f in FAMS:
        rs = load(f, L)
        if not rs: continue
        def g(k):
            return safe(lambda x: x["sup"][k][0] if isinstance(x["sup"][k], list) else x["sup"][k])
        print(f"| {f} | " + " | ".join(ms([layer_avg(r, g(k)) for r in rs]) for k in SK) + " |")

print("\n## D. per-layer prefix region at L4096: massF-massE (bias-induced mass shift) and corr(B,E)")
for f in FAMS:
    rs = load(f, 4096)
    if not rs: continue
    ls = sorted(rs[0]["layers"], key=int)
    dm = [ms([r["layers"][l]["region"]["prefix"]["massF"] - r["layers"][l]["region"]["prefix"]["massE"] for r in rs]) for l in ls]
    cb = [ms([r["layers"][l]["region"]["prefix"]["corrBE"] for r in rs]) for l in ls]
    print(f"{f:9s} dmass " + " ".join(dm))
    print(f"{f:9s} corrBE " + " ".join(cb))

print("\n## E. prefix profile at L4096, layer-avg, 16-token buckets over keys 0..511 (B and E)")
for f in FAMS:
    rs = load(f, 4096)
    if not rs: continue
    for k in ("B", "E", "pF"):
        arr = np.mean([[np.mean([r["layers"][l]["profile"][k][b] for l in r["layers"]]) for b in range(32)] for r in rs], 0)
        print(f"{f:9s} {k:2s} " + " ".join(f"{x:+.2f}" for x in arr[::2]))

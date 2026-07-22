#!/usr/bin/env python3
"""
PAT-234 dose-response: does centering's F1 benefit (C - A) track its mechanistic
attention effect (softmax-visible attn_KL amplification) across K=1 scales?

x-axis = centering attn_KL ratio (center-ON / center-OFF), from pipeline probe 527841 (L512).
y-axis = dF1 = F1(center-ON C) - F1(center-OFF A), at a chosen eval length.

Reads eval_f1 from results_uniform/<name>/L<L>/all_results.json.
Missing results are marked pending (script is safe to run partially).
Usage: aggregate_pat234_dose_response.py [L]   (default L=4096)
"""
import json, sys
from pathlib import Path

L = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
RU = Path("/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/results_uniform")
OUT = Path("/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat234_scale_card")

# scale -> (C model name, A model name, attn_KL center-OFF, attn_KL center-ON)  [probe 527841, L512, shift ON]
SCALES = {
    "rho1":     ("pat234_K1_me0_C_s42_ckpt15000",  "pat225_k1sc_S1_me0_s42",  3.534e-06, 9.395e-03),
    "rho16":    ("pat234_K1_me8_C_s42_ckpt15000",  "pat225_k1sc_S1_me8_s42",  6.044e-02, 3.583e-02),
    "rho128":   ("pat234_K1_me14_C_s42_ckpt15000", "pat225_k1sc_S1_s42",      2.287e-02, 1.816e-02),
    "rho1024":  ("pat234_K1_me20_C_s42_ckpt15000", "pat225_k1sc_S1_me20_s42", 6.639e-03, 1.294e-02),
    "rho16384": ("pat234_K1_me28_C_s42_ckpt15000", "pat225_k1sc_S1_me28_s42", 7.260e-04, 1.402e-02),
}


def f1_of(name):
    f = RU / name / f"L{L}" / "all_results.json"
    if not f.is_file():
        return None
    try:
        return float(json.load(open(f))["eval_f1"])
    except Exception:
        return None


def main():
    rows = []
    print(f"=== PAT-234 dose-response (L={L}) ===")
    print(f"{'scale':<9} {'A_F1':>8} {'C_F1':>8} {'dF1':>8} | {'KL_off':>9} {'KL_on':>9} {'ratio':>7}")
    for sc, (cn, an, kl_off, kl_on) in SCALES.items():
        a, c = f1_of(an), f1_of(cn)
        ratio = kl_on / max(kl_off, 1e-12)
        df1 = (c - a) if (a is not None and c is not None) else None
        astr = f"{a:.4f}" if a is not None else "pending"
        cstr = f"{c:.4f}" if c is not None else "pending"
        dstr = f"{df1:+.4f}" if df1 is not None else "  --  "
        print(f"{sc:<9} {astr:>8} {cstr:>8} {dstr:>8} | {kl_off:>9.2e} {kl_on:>9.2e} {ratio:>6.1f}x")
        if df1 is not None:
            rows.append((sc, ratio, df1))

    if len(rows) >= 2:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
            xs = np.array([r[1] for r in rows]); ys = np.array([r[2] for r in rows])
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.axhline(0, color="gray", lw=1, ls="--")
            ax.axvline(1, color="gray", lw=0.8, ls=":")
            ax.scatter(xs, ys, s=90, c=["#d62728" if y > 0 else "#1f77b4" for y in ys], zorder=3)
            for sc, x, y in rows:
                ax.annotate(sc, (x, y), textcoords="offset points", xytext=(6, 5), fontsize=9)
            ax.set_xscale("log")
            ax.set_xlabel("centering attention effect  (attn_KL ratio  center-ON / center-OFF)")
            ax.set_ylabel(f"dF1 = F1(center-ON) - F1(center-OFF)   @ L{L}")
            ax.set_title(f"PAT-234 dose-response (K=1, L{L}): does F1 benefit track the attention effect?")
            ax.grid(True, alpha=0.3)
            out = OUT / f"dose_response_L{L}.png"
            fig.tight_layout(); fig.savefig(out, dpi=140)
            print(f"\nsaved figure: {out}  ({len(rows)}/5 scales ready)")
        except Exception as e:
            print(f"(plot skipped: {e})")
    else:
        print(f"\n(only {len(rows)}/5 scales ready — need >=2 for a plot)")


if __name__ == "__main__":
    main()

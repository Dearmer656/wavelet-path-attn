#!/usr/bin/env python3
"""Compare fixed-rho vs L_test-adaptive-rho wavelet basis shape.

Reproduces the real forward-pass convention (causal window ending at the query,
full-T non-causal RMS normalization, matching `_rms_norm_wavelet_basis`).

Three conditions:
  A) T=512  (L_train), rho=256           -- training reference
  B) T=2048 (L_test),  rho=256  (fixed)   -- naive extrapolation, alpha shrinks
  C) T=2048 (L_test),  rho=1024 (adaptive)-- alpha kept == training (rho scales with T)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 1e-6


def ricker(u):
    return (1.0 - u**2) * np.exp(-0.5 * u**2)


def rms_normalize(x, eps=EPS):
    return x / np.sqrt(np.mean(x**2) + eps)


def build_basis(T, rho):
    k = np.arange(T, dtype=np.float64)
    beta = float(T - 1)  # query anchored at the last position (full causal window available)
    u = (k - beta) / rho
    return rms_normalize(ricker(u)), k - beta  # also return delta = k - q


conditions = [
    ("A) T=512 (L_train), rho=256 (training)", 512, 256, "tab:blue", "-"),
    ("B) T=2048 (L_test), rho=256 (fixed)", 2048, 256, "tab:red", "-"),
    ("C) T=2048 (L_test), rho=1024 (adaptive, alpha=0.5)", 2048, 1024, "tab:green", "--"),
]

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

# Left panel: absolute token distance from query (delta = k - q)
ax = axes[0]
for label, T, rho, color, ls in conditions:
    basis, delta = build_basis(T, rho)
    ax.plot(delta, basis, label=label, color=color, linestyle=ls, linewidth=1.8)
ax.set_xlim(-2200, 100)
ax.set_xlabel("absolute distance from query, δ = k - q  (tokens)")
ax.set_ylabel("RMS-normalized wavelet bias")
ax.set_title("Absolute token axis")
ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.25)

# Right panel: relative position (delta / T) -- this is where "adaptive recovers
# the training-time relative shape" becomes visually obvious
ax = axes[1]
for label, T, rho, color, ls in conditions:
    basis, delta = build_basis(T, rho)
    ax.plot(delta / T, basis, label=label, color=color, linestyle=ls, linewidth=1.8)
ax.set_xlim(-1.05, 0.05)
ax.set_xlabel("relative distance from query, δ/T  (fraction of context window)")
ax.set_ylabel("RMS-normalized wavelet bias")
ax.set_title("Relative (window-normalized) axis — this is the key comparison")
ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.25)

fig.suptitle(
    "Fixed rho vs L_test-adaptive rho: does extrapolation preserve the training-time wavelet shape?\n"
    "alpha = rho/T:  A (train) = 0.500  |  B (fixed @ L_test) = 0.125  |  C (adaptive @ L_test) = 0.500 (matches A)",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.90])
out_path = "/tmp/claude-40779/-project-nlp-work5-hongyu-s-transformers/9cab3a6c-777f-4eed-98b4-f816a476f500/scratchpad/fixed_vs_adaptive_rho.png"
fig.savefig(out_path, dpi=150)
print(f"Wrote {out_path}")

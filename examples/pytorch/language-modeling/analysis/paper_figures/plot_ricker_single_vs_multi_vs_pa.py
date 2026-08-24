"""Paper figure: Ricker basis, PA-only vs K1 single-scale vs K3 multi-scale,
small model (GPT-2 small, 12 layers), HotpotQA-Long F1 at L=2048, all 3-seed.

Excludes K1 rho=256 (only 1 seed, checkpoint never trained for s43/s44 --
permanent gap) and K3 static-learned router variant (per author decision, keep
the comparison to the two endpoints: zero-learned fixed ratio vs fully learned
query-conditioned mixing).
"""
import matplotlib.pyplot as plt
import numpy as np

OUT_PDF = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/analysis/paper_figures/ricker_single_vs_multi_vs_pa.pdf"

PA_ONLY = {42: 0.7083, 43: 0.7146, 44: 0.7112}
K1_RHO128 = {42: 0.7231, 43: 0.7294, 44: 0.7191}
K1_RHO384 = {42: 0.7229, 43: 0.7110, 44: 0.7260}
K3_FIXED_RATIO = {42: 0.7256, 43: 0.7265, 44: 0.7262}
K3_FULLY_LEARNED = {42: 0.7278, 43: 0.7303, 44: 0.7268}


def mean_std(d):
    vals = list(d.values())
    return float(np.mean(vals)), float(np.std(vals, ddof=1))


def main():
    groups = [
        ("PA-only", PA_ONLY, "#4C72B0"),
        ("K1\nρ=128", K1_RHO128, "#DD8452"),
        ("K1\nρ=384", K1_RHO384, "#DD8452"),
        ("K3 fixed\nratio", K3_FIXED_RATIO, "#55A868"),
        ("K3 fully\nlearned", K3_FULLY_LEARNED, "#2E7D32"),
    ]

    labels = [g[0] for g in groups]
    means = []
    stds = []
    colors = [g[2] for g in groups]
    for _, d, _ in groups:
        m, s = mean_std(d)
        means.append(m)
        stds.append(s)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, width=0.6, zorder=3)

    for xi, m in zip(x, means):
        ax.annotate(f"{m:.4f}", (xi, m), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=9, zorder=4)

    pa_mean = means[0]
    ax.axhline(pa_mean, color="#4C72B0", linestyle="--", linewidth=1.0, alpha=0.6, zorder=2)
    ax.axvline(0.5, color="#888888", linewidth=0.6, linestyle=":", zorder=1)
    ax.axvline(2.5, color="#888888", linewidth=0.6, linestyle=":", zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("HotpotQA-Long F1 (L=2048)", fontsize=11)
    ax.set_title("Ricker basis: PA-only vs single-scale vs multi-scale (small, 3-seed)", fontsize=11)
    ax.set_ylim(0.695, 0.735)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_PDF)
    print(f"Saved: {OUT_PDF}")
    for label, m, s in zip(labels, means, stds):
        print(f"{label.replace(chr(10), ' ')}: mean={m:.4f} std={s:.4f} n=3")


if __name__ == "__main__":
    main()

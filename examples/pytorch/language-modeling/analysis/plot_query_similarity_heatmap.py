import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/analysis/query_similarity_out"
L_TRAIN = 512

nope = np.load(os.path.join(OUT_DIR, "heatmap_NoPE.npy"))
path = np.load(os.path.join(OUT_DIR, "heatmap_PaTH.npy"))
with open(os.path.join(OUT_DIR, "summary.json")) as f:
    summary = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, mat, title in zip(axes, [path, nope], ["PaTH ($q^{corr}$)", "NoPE (hidden state)"]):
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis", origin="lower")
    ax.axvline(L_TRAIN, color="red", linewidth=1, linestyle="--")
    ax.axhline(L_TRAIN, color="red", linewidth=1, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("query position")
    ax.set_ylabel("query position")
fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04, label="cosine similarity")
fig.suptitle(
    "Router conditioning-feature similarity across query positions (layer-avg, N=3 example avg)\n"
    f"red dashed line = $L_{{train}}$={L_TRAIN}",
    fontsize=10,
)
out_path = os.path.join(OUT_DIR, "query_similarity_heatmap.pdf")
fig.savefig(out_path, bbox_inches="tight", dpi=150)
out_png = os.path.join(OUT_DIR, "query_similarity_heatmap.png")
fig.savefig(out_png, bbox_inches="tight", dpi=150)
print(f"saved: {out_path}")
print(f"saved: {out_png}")

print("\nLaTeX table:")
r_path = summary["PaTH"]
r_nope = summary["NoPE"]
print(r"""\begin{table}[t]
\centering
\caption{Mean off-diagonal cosine similarity of the QWAB router's conditioning feature across query positions, in-distribution ($\le L_{\text{train}}$) vs.\ extrapolation ($>L_{\text{train}}$). PaTH's $q^{\text{corr}}$ feature stays position-differentiated under extrapolation; NoPE's plain hidden state collapses toward positional indistinguishability. $N=100$ examples $\times$ 12 layers = 1200 (example, layer) pairs per cell.}
\label{tab:query-similarity-nope-vs-path}
\begin{tabular}{lccc}
\toprule
Model & In-distribution ($\le 512$) & Extrapolation ($>512$) & $\Delta$ \\
\midrule""")
print(f"PaTH ($q^{{corr}}$) & {r_path['in_mean']:.4f} $\\pm$ {r_path['in_std']:.4f} & {r_path['extrap_mean']:.4f} $\\pm$ {r_path['extrap_std']:.4f} & +{r_path['extrap_mean']-r_path['in_mean']:.4f} \\\\")
print(f"NoPE (hidden state) & {r_nope['in_mean']:.4f} $\\pm$ {r_nope['in_std']:.4f} & {r_nope['extrap_mean']:.4f} $\\pm$ {r_nope['extrap_std']:.4f} & \\textbf{{+{r_nope['extrap_mean']-r_nope['in_mean']:.4f}}} \\\\")
print(r"""\bottomrule
\end{tabular}
\end{table}""")

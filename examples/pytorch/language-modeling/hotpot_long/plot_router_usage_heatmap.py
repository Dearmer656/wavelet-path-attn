#!/usr/bin/env python3
"""Layer x scale heatmap of K3 router usage (3-seed mean), small vs medium,
hotpot vs xsum, at each model's train length L512. Reads the CSVs written
by dump_router_usage.py under analysis_outputs/router_usage/.
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
USAGE_DIR = ROOT / "analysis_outputs" / "router_usage"
OUT_DIR = ROOT / "analysis_outputs" / "router_usage" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLS = ["null", "scale0", "scale1", "scale2"]


def load_mean_matrix(model_tag: str, task: str, seq_len: str, seeds=("42", "43", "44")):
    prefix = f"{model_tag}_xsum" if task == "xsum" else model_tag
    sums = defaultdict(lambda: np.zeros(4, dtype=np.float64))
    counts = defaultdict(int)
    n_layers = None
    for s in seeds:
        path = USAGE_DIR / f"{prefix}_s{s}.csv"
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row["seq_len"] != seq_len:
                    continue
                layer = int(row["layer"])
                n_layers = max(n_layers or 0, int(row["n_layers"]))
                vals = np.array([
                    float(row["null_usage"]),
                    float(row["scale0_usage"]),
                    float(row["scale1_usage"]),
                    float(row["scale2_usage"]),
                ])
                sums[layer] += vals
                counts[layer] += 1
    mat = np.zeros((n_layers, 4), dtype=np.float64)
    for layer in range(n_layers):
        mat[layer] = sums[layer] / max(counts[layer], 1)
    return mat


def plot_panel(ax, mat, title, vmax):
    im = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=0.0, vmax=vmax)
    ax.set_xticks(range(4))
    ax.set_xticklabels(COLS)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels([str(i) for i in range(mat.shape[0])])
    ax.set_xlabel("gate")
    ax.set_ylabel("layer")
    ax.set_title(title, fontsize=10)
    for i in range(mat.shape[0]):
        for j in range(4):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                     color="white" if mat[i, j] < 0.5 * vmax else "black", fontsize=6)
    return im


def main():
    panels = [
        ("small", "hotpot", "512", "Small, HotpotQA, L512"),
        ("medium", "hotpot", "512", "Medium, HotpotQA, L512"),
        ("small", "xsum", "512", "Small, XSum, L512"),
        ("medium", "xsum", "512", "Medium, XSum, L512"),
    ]
    _plot_set(panels, "router_usage_heatmap_small_vs_medium_sharedscale.png",
              "K3 router usage (3-seed mean) by layer x scale, train length L512")


def _plot_set(panels, out_name, suptitle_prefix):
    mats = [load_mean_matrix(model_tag, task, seq_len) for model_tag, task, seq_len, _ in panels]
    shared_vmax = float(max(m.max() for m in mats))

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 9))
    if len(panels) == 1:
        axes = [axes]
    im = None
    for ax, mat, (model_tag, task, seq_len, title) in zip(axes, mats, panels):
        im = plot_panel(ax, mat, title, vmax=shared_vmax)
    fig.colorbar(im, ax=axes, shrink=0.7, label="usage (shared scale across all panels)")
    fig.suptitle(f"{suptitle_prefix} -- shared color scale (vmax={shared_vmax:.2f})", fontsize=13)
    out_path = OUT_DIR / out_name
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"wrote {out_path}")


def main_long():
    # Extended lengths: both small and medium now have real hotpot L4096 data
    # (medium via cross-node 2xGPU head-parallel on elm72+elm73, since dense
    # pytorch single-GPU OOMs medium at L4096) -- apples-to-apples comparison.
    # No medium L16384 panel: the front-placed citation dataset
    # (data/hotpot_long_dev.jsonl) tops out at L4096, and L8192/L16384 were
    # confirmed OOM even with 2xGPU head-parallel (fp32 and bf16 both) -- see
    # PAT-253 notes.
    panels = [
        ("small", "hotpot", "4096", "Small, HotpotQA, L4096"),
        ("medium", "hotpot", "4096", "Medium, HotpotQA, L4096 (cross-node 2xGPU head-parallel)"),
    ]
    _plot_set(panels, "router_usage_heatmap_small_vs_medium_longlen.png",
              "K3 router usage (3-seed mean) by layer x scale, extended lengths")


if __name__ == "__main__":
    import sys
    if "--long" in sys.argv:
        main_long()
    else:
        main()

import matplotlib.pyplot as plt

def make_chart(title, ylabel, x_labels, series, ylim, yticks, out_path, source_note=None):
    """series: list of tuples, in increasing order of optionality:
    (name, color, linestyle, linewidth, markersize, values)
    (name, color, linestyle, linewidth, markersize, values, marker_shape)
    (name, color, linestyle, linewidth, markersize, values, marker_shape, stds)
    `stds` (if given) is a list matching `values`, drawn as error bars.
    """
    x = list(range(len(x_labels)))

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=200)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    for entry in series:
        name, color, ls, lw, ms = entry[0:5]
        vals = entry[5]
        shape = entry[6] if len(entry) >= 7 else "o"
        stds = entry[7] if len(entry) >= 8 else None
        zorder = 3 if name.startswith("K3") else 2

        if stds is not None:
            ax.errorbar(x, vals, yerr=stds, color=color, linestyle=ls, linewidth=lw,
                        marker=shape, markersize=ms, markerfacecolor=color,
                        markeredgecolor="#ffffff", markeredgewidth=1.2,
                        label=name, zorder=zorder, capsize=4, capthick=1.3,
                        elinewidth=1.3, ecolor=color)
        else:
            ax.plot(x, vals, color=color, linestyle=ls, linewidth=lw,
                     marker=shape, markersize=ms, markerfacecolor=color,
                     markeredgecolor="#ffffff", markeredgewidth=1.2,
                     label=name, zorder=zorder)

    for i in range(len(x_labels)):
        at_len = [(e[0], e[1], e[5][i]) for e in series]
        max_name, max_color, max_val = max(at_len, key=lambda t: t[2])
        min_name, min_color, min_val = min(at_len, key=lambda t: t[2])

        ax.annotate(f"{max_val:.4f}", xy=(i, max_val), xytext=(0, 14),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=12, color=max_color, fontweight="bold")
        ax.annotate(f"{min_val:.4f}", xy=(i, min_val), xytext=(0, -14),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=12, color=min_color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=15)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_ylabel(ylabel, fontsize=16, color="#1a1d24")
    ax.set_xlabel("Evaluation Length", fontsize=16, color="#1a1d24", labelpad=10)
    ax.set_title(title, fontsize=19, color="#1a1d24", pad=16)

    ax.grid(True, color="#e8eaf0", linewidth=1, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#000000")
        spine.set_linewidth(1.3)

    ax.set_xlim(-0.15, len(x_labels) - 1 + 0.15)
    ax.tick_params(colors="#1a1d24", labelsize=14)
    ax.legend(loc="lower left", frameon=False, fontsize=13.5)

    plt.tight_layout()
    plt.savefig(out_path, transparent=True)
    print("saved", out_path)

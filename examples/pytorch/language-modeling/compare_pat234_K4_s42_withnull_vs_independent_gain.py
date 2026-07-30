import math
from pathlib import Path
from safetensors import safe_open
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = {
    "with_null (sqrtnorm)": "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat234_scale_card/K4_me8_16_20_24_noC1_s42_sqrtnorm",
    "independent_scales (independent_sqrtnorm)": "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat234_scale_card/K4_me8_16_20_24_noC1_s42_independent_sqrtnorm",
}
LABELS = list(RUNS.keys())
STEPS = [2500, 5000, 7500, 10000, 12500, 15000]
NUM_LAYERS = 12
OUT_DIR = Path("/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/analysis/k4_s42_with_null_vs_independent")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def softplus(x):
    return math.log1p(math.exp(-abs(x))) + max(x, 0)


def load_keys(ckpt_dir):
    path = Path(ckpt_dir) / "model.safetensors"
    out = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        keys = set(f.keys())
        for layer in range(NUM_LAYERS):
            prefix = f"transformer.h.{layer}.attn.core."
            a_key = prefix + "wavelet_logit_bias_a"
            k1_key = prefix + "wavelet_k1_gain"
            router_w_key = prefix + "wavelet_ctx_router.weight"
            if a_key in keys:
                out.setdefault("a", {})[layer] = f.get_tensor(a_key).float().mean().item()
            if k1_key in keys:
                out.setdefault("k1_gain", {})[layer] = f.get_tensor(k1_key).float().mean().item()
            if router_w_key in keys:
                out.setdefault("router_w_rms", {})[layer] = f.get_tensor(router_w_key).float().pow(2).mean().sqrt().item()
    return out


results = {}
for label, run_dir in RUNS.items():
    results[label] = {}
    for step in STEPS:
        ckpt = Path(run_dir) / f"checkpoint-{step}"
        if ckpt.exists():
            results[label][step] = load_keys(ckpt)

# ---------- text report ----------
report = []
report.append("=== Aggregate (mean over 12 layers) wavelet_logit_bias_a and g_layer=softplus(a) trajectory ===")
report.append(f"{'step':>7} | {'with_null a':>12} {'with_null g':>12} | {'indep a':>12} {'indep g':>12}")
agg_g = {label: [] for label in LABELS}
agg_a = {label: [] for label in LABELS}
for step in STEPS:
    parts = []
    for label in LABELS:
        a_dict = results[label].get(step, {}).get("a", {})
        if a_dict:
            a_mean = sum(a_dict.values()) / len(a_dict)
            g_mean = sum(softplus(v) for v in a_dict.values()) / len(a_dict)
        else:
            a_mean, g_mean = float("nan"), float("nan")
        agg_a[label].append(a_mean)
        agg_g[label].append(g_mean)
        parts.append(f"{a_mean:>12.4f} {g_mean:>12.4f}")
    report.append(f"{step:>7} | " + " | ".join(parts))

report.append("")
report.append("=== Per-layer wavelet_logit_bias_a / g_layer at checkpoint-15000 ===")
report.append(f"{'layer':>5} | {'with_null a':>12} {'with_null g':>12} | {'indep a':>12} {'indep g':>12} | {'diff g (indep-null)':>20}")
per_layer_g_final = {label: [] for label in LABELS}
for layer in range(NUM_LAYERS):
    parts = []
    gvals = {}
    for label in LABELS:
        a_dict = results[label].get(15000, {}).get("a", {})
        a_v = a_dict.get(layer)
        if a_v is None:
            parts.append(f"{'NA':>12} {'NA':>12}")
            gvals[label] = float("nan")
        else:
            g_v = softplus(a_v)
            parts.append(f"{a_v:>12.4f} {g_v:>12.4f}")
            gvals[label] = g_v
        per_layer_g_final[label].append(gvals[label])
    diff = gvals[LABELS[1]] - gvals[LABELS[0]]
    report.append(f"{layer:>5} | " + " | ".join(parts) + f" | {diff:>20.4f}")

report.append("")
report.append("=== router weight RMS (proxy for router/gate parameter growth), mean over layers ===")
report.append(f"{'step':>7} | {'with_null':>12} | {'indep':>12}")
agg_router = {label: [] for label in LABELS}
for step in STEPS:
    parts = []
    for label in LABELS:
        rw = results[label].get(step, {}).get("router_w_rms", {})
        v = sum(rw.values()) / len(rw) if rw else float("nan")
        agg_router[label].append(v)
        parts.append(f"{v:>12.6f}")
    report.append(f"{step:>7} | " + " | ".join(parts))

report.append("")
report.append("=== wavelet_k1_gain (if present/nonzero) at checkpoint-15000 ===")
for label in LABELS:
    k1 = results[label].get(15000, {}).get("k1_gain", {})
    if k1:
        report.append(f"{label}: mean over layers = {sum(k1.values())/len(k1):.6f}, per-layer: {k1}")
    else:
        report.append(f"{label}: no wavelet_k1_gain key found")

report_text = "\n".join(report)
print(report_text)
(OUT_DIR / "report.txt").write_text(report_text)

# ---------- plots ----------
short_labels = {LABELS[0]: "with_null (F1=0.6973)", LABELS[1]: "independent_scales (F1=0.6244)"}

# 1. g_layer trajectory (mean over layers) vs step
fig, ax = plt.subplots(figsize=(7, 5))
for label in LABELS:
    ax.plot(STEPS, agg_g[label], marker="o", label=short_labels[label])
ax.set_xlabel("training step")
ax.set_ylabel("mean g_layer = softplus(a_l) over 12 layers")
ax.set_title("K4 seed42: QWAB layer-gain trajectory\nwith_null vs independent_scales")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "01_g_layer_trajectory.png", dpi=150)
plt.close(fig)

# 2. Difference in aggregate g_layer trajectory (independent - with_null)
diff_traj = [g1 - g0 for g0, g1 in zip(agg_g[LABELS[0]], agg_g[LABELS[1]])]
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(STEPS, diff_traj, marker="o", color="crimson")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("training step")
ax.set_ylabel("Δ mean g_layer (independent − with_null)")
ax.set_title("K4 seed42: layer-gain gap over training\n(negative = independent_scales has smaller amplitude)")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "02_g_layer_diff_trajectory.png", dpi=150)
plt.close(fig)

# 3. Per-layer g_layer at final checkpoint, grouped bars
x = list(range(NUM_LAYERS))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar([i - width / 2 for i in x], per_layer_g_final[LABELS[0]], width, label=short_labels[LABELS[0]])
ax.bar([i + width / 2 for i in x], per_layer_g_final[LABELS[1]], width, label=short_labels[LABELS[1]])
ax.set_xlabel("layer index")
ax.set_ylabel("g_layer at checkpoint-15000")
ax.set_title("K4 seed42: per-layer final g_layer comparison")
ax.set_xticks(x)
ax.legend()
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUT_DIR / "03_per_layer_g_final.png", dpi=150)
plt.close(fig)

# 4. Per-layer difference bar chart at final checkpoint
diff_per_layer = [g1 - g0 for g0, g1 in zip(per_layer_g_final[LABELS[0]], per_layer_g_final[LABELS[1]])]
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["tab:red" if d < 0 else "tab:green" for d in diff_per_layer]
ax.bar(x, diff_per_layer, color=colors)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("layer index")
ax.set_ylabel("Δ g_layer (independent − with_null)")
ax.set_title("K4 seed42: per-layer amplitude gap at checkpoint-15000")
ax.set_xticks(x)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUT_DIR / "04_per_layer_g_diff.png", dpi=150)
plt.close(fig)

# 5. router weight RMS trajectory
fig, ax = plt.subplots(figsize=(7, 5))
for label in LABELS:
    ax.plot(STEPS, agg_router[label], marker="o", label=short_labels[label])
ax.set_xlabel("training step")
ax.set_ylabel("mean wavelet_ctx_router.weight RMS over 12 layers")
ax.set_title("K4 seed42: router weight RMS trajectory")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "05_router_weight_rms_trajectory.png", dpi=150)
plt.close(fig)

print(f"\nPlots + report written to {OUT_DIR}")

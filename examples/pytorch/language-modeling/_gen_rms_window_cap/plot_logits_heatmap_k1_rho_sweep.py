#!/usr/bin/env python3
"""Compare K=1 Ricker QWAB models at 0.25/0.5/0.75 * L_train scales."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from plot_logits_heatmap_pa_vs_qwab import (
    LAYERS_TO_PLOT,
    PA_ONLY_CFG_PATH,
    PA_ONLY_CHECKPOINT,
    QUERY_ROWS_TO_PLOT,
    T,
    _corr,
    _rms,
    _vector_from_diag,
    build_batch,
    load_model,
    run_and_capture,
)


ROOT = Path("/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling")
OUT_DIR = ROOT / "_gen_rms_window_cap"
L_TRAIN = 512
MODELS = [
    {
        "tag": "rho128_0p25Ltrain",
        "rho_scale": 128.0,
        "fraction": 0.25,
        "run": "K1_L512_me14_rho128_ricker_s42",
    },
    {
        "tag": "rho256_0p50Ltrain",
        "rho_scale": 256.0,
        "fraction": 0.50,
        "run": "K1_L512_me16_rho256_ricker_s42",
    },
    {
        "tag": "rho384_0p75Ltrain",
        "rho_scale": 384.0,
        "fraction": 0.75,
        "run": "K1_L512_me17p1699_rho384_ricker_s42",
    },
]


def _model_paths(run_name):
    run_dir = ROOT / "runs" / "pat244_dual_temp" / run_name
    return str(run_dir / "checkpoint-15000"), str(run_dir / "supply_model.cfg")


def _causal_values(matrix):
    mask = np.tri(matrix.shape[0], matrix.shape[1], dtype=bool)
    return matrix[mask]


def plot_heatmap(model_info, pa_baseline, pa_component, full_component):
    fig, axes = plt.subplots(len(LAYERS_TO_PLOT), 5, figsize=(26, 4.5 * len(LAYERS_TO_PLOT)))
    for row, layer in enumerate(LAYERS_TO_PLOT):
        pa_base = pa_baseline[layer]
        pa_comp = pa_component[layer]
        full = full_component[layer]
        bias = full - pa_comp
        imprint = pa_comp - pa_base
        future = np.triu(np.ones_like(pa_base, dtype=bool), k=1)
        displays = [
            np.where(future, np.nan, pa_base),
            np.where(future, np.nan, pa_comp),
            np.where(future, np.nan, full),
        ]
        vmin = min(float(np.nanmin(x)) for x in displays)
        vmax = max(float(np.nanmax(x)) for x in displays)
        titles = ["trained PA-only", "K1 checkpoint PA", "K1 full (PA+QWAB)"]
        for col, (value, title) in enumerate(zip(displays, titles)):
            image = axes[row, col].imshow(value, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
            axes[row, col].set_title(f"layer {layer}: {title}")
            plt.colorbar(image, ax=axes[row, col], fraction=0.046)

        for col, (delta, title) in enumerate(
            [(bias, "direct QWAB bias"), (imprint, "training imprint")], start=3
        ):
            value = np.where(future, np.nan, delta)
            dmax = max(float(np.nanmax(np.abs(value))), 1e-8)
            image = axes[row, col].imshow(value, cmap="PuOr", vmin=-dmax, vmax=dmax, aspect="auto")
            axes[row, col].set_title(f"layer {layer}: {title}")
            plt.colorbar(image, ax=axes[row, col], fraction=0.046)

        for ax in axes[row]:
            ax.set_xlabel("key position")
            ax.set_ylabel("query position")

    fig.suptitle(
        f"K=1 Ricker QWAB: rho(scale)={model_info['rho_scale']:.0f} "
        f"= {model_info['fraction']:.2f} * L_train, L_train={L_TRAIN}, L_test={T}",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = OUT_DIR / f"logits_heatmap_k1_{model_info['tag']}_L{T}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_row_sweep(layer, results):
    query_rows = [q for q in QUERY_ROWS_TO_PLOT if 0 <= q < T]
    fig, axes = plt.subplots(
        len(query_rows),
        2 * len(MODELS),
        figsize=(32, 4.2 * len(query_rows)),
        squeeze=False,
    )
    for model_idx, model_info in enumerate(MODELS):
        result = results[model_info["tag"]]
        pa_base = result["pa_baseline"][layer]
        pa_comp = result["pa_component"][layer]
        full = result["full_component"][layer]
        bias = full - pa_comp
        diag = result["diagnostics"][layer]
        g0 = _vector_from_diag(diag["g0"], T)
        shift_rho = _vector_from_diag(diag["rho"], T)
        beta = _vector_from_diag(diag["beta"], T)

        for row_idx, q in enumerate(query_rows):
            keys = np.arange(q + 1)
            base_row = pa_base[q, : q + 1]
            pa_row = pa_comp[q, : q + 1]
            full_row = full[q, : q + 1]
            bias_row = bias[q, : q + 1]
            g0_q = float(g0[q])
            conditional = bias_row / max(abs(g0_q), 1e-8)
            centered_bias = bias_row - float(np.mean(bias_row))

            raw_ax = axes[row_idx, 2 * model_idx]
            raw_ax.plot(keys, base_row, color="0.60", linewidth=0.65, label="PA-only")
            raw_ax.plot(keys, pa_row, color="#1d3557", linewidth=0.72, label="K1 PA")
            raw_ax.plot(keys, full_row, color="#e63946", linewidth=0.72, alpha=0.85, label="K1 full")
            raw_ax.set_title(
                f"rho={model_info['rho_scale']:.0f} ({model_info['fraction']:.2f}L), q={q}\n"
                f"raw logits; bias/PA RMS={_rms(bias_row) / max(_rms(pa_row), 1e-12):.3f}"
            )
            raw_ax.set_xlabel("visible key position k")
            raw_ax.set_ylabel("mean-head logit")
            raw_ax.legend(loc="best", fontsize=7)

            wav_ax = axes[row_idx, 2 * model_idx + 1]
            wav_ax.plot(keys, conditional, color="#457b9d", linewidth=1.0, label="bias_eff / g0")
            wav_ax.plot(keys, bias_row, color="#e63946", linewidth=1.0, label="bias_eff")
            wav_ax.plot(
                keys,
                centered_bias,
                color="#f4a261",
                linewidth=0.9,
                linestyle="--",
                label="bias_eff - row mean",
            )
            wav_ax.axhline(0.0, color="0.4", linewidth=0.5)
            wav_ax.set_title(
                f"g0={g0_q:.3f}, shift-rho={shift_rho[q]:.3f}, beta_unit={beta[q]:.3f}\n"
                f"bias RMS={_rms(bias_row):.3f}, softmax-effective centered RMS={_rms(centered_bias):.3f}"
            )
            wav_ax.set_xlabel("visible key position k")
            wav_ax.set_ylabel("wavelet logit bias")
            wav_ax.legend(loc="best", fontsize=7)

    fig.suptitle(
        f"Layer {layer}: K=1 row-level comparison across Ricker rho(scale), L_test={T}\n"
        "Each rho pair shows raw logits and the exact additive QWAB wavelet row",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = OUT_DIR / f"logits_rows_k1_rho_sweep_layer{layer}_L{T}.png"
    fig.savefig(out_path, dpi=135)
    plt.close(fig)
    print(f"Wrote {out_path}")


def write_metrics(results):
    rows = []
    for model_info in MODELS:
        result = results[model_info["tag"]]
        for layer in LAYERS_TO_PLOT:
            pa_base = result["pa_baseline"][layer]
            pa_comp = result["pa_component"][layer]
            full = result["full_component"][layer]
            bias = full - pa_comp
            imprint = pa_comp - pa_base
            diag = result["diagnostics"][layer]
            g0 = _vector_from_diag(diag["g0"], T)
            full_bias_rms = np.sqrt(np.mean(bias**2, axis=1))
            causal_bias = _causal_values(bias)
            causal_centered_bias = np.concatenate(
                [bias[q, : q + 1] - float(np.mean(bias[q, : q + 1])) for q in range(T)]
            )
            causal_pa = _causal_values(pa_comp)
            causal_imprint = _causal_values(imprint)
            rows.append({
                "scope": "layer_causal_all",
                "rho_scale": model_info["rho_scale"],
                "rho_over_Ltrain": model_info["fraction"],
                "layer": layer,
                "query": "all",
                "g0": float(np.mean(g0)),
                "pa_rms": _rms(causal_pa),
                "training_imprint_rms": _rms(causal_imprint),
                "wavelet_bias_rms": _rms(causal_bias),
                "row_centered_wavelet_bias_rms": _rms(causal_centered_bias),
                "centered_over_raw_bias_rms": _rms(causal_centered_bias) / max(_rms(causal_bias), 1e-12),
                "wavelet_over_pa_rms": _rms(causal_bias) / max(_rms(causal_pa), 1e-12),
                "wavelet_over_training_imprint_rms": _rms(causal_bias) / max(_rms(causal_imprint), 1e-12),
                "corr_wavelet_pa": _corr(causal_bias, causal_pa),
                "corr_g0_full_context_bias_rms": _corr(g0, full_bias_rms),
                "mean_abs_full_context_bias_rms_minus_g0": float(np.mean(np.abs(full_bias_rms - g0))),
            })
            for q in QUERY_ROWS_TO_PLOT:
                bias_row = bias[q, : q + 1]
                centered_bias_row = bias_row - float(np.mean(bias_row))
                pa_row = pa_comp[q, : q + 1]
                imprint_row = imprint[q, : q + 1]
                rows.append({
                    "scope": "query_row_causal",
                    "rho_scale": model_info["rho_scale"],
                    "rho_over_Ltrain": model_info["fraction"],
                    "layer": layer,
                    "query": q,
                    "g0": float(g0[q]),
                    "pa_rms": _rms(pa_row),
                    "training_imprint_rms": _rms(imprint_row),
                    "wavelet_bias_rms": _rms(bias_row),
                    "row_centered_wavelet_bias_rms": _rms(centered_bias_row),
                    "centered_over_raw_bias_rms": _rms(centered_bias_row) / max(_rms(bias_row), 1e-12),
                    "wavelet_over_pa_rms": _rms(bias_row) / max(_rms(pa_row), 1e-12),
                    "wavelet_over_training_imprint_rms": _rms(bias_row) / max(_rms(imprint_row), 1e-12),
                    "corr_wavelet_pa": _corr(bias_row, pa_row),
                    "corr_g0_full_context_bias_rms": "",
                    "mean_abs_full_context_bias_rms_minus_g0": "",
                })

    metrics_path = OUT_DIR / f"logits_k1_rho_sweep_L{T}.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {metrics_path}")


def plot_controlled_time_domain():
    """Show scale alone with g0=1 and shift=0, including softmax-invariant mean removal."""
    query_rows = [q for q in QUERY_ROWS_TO_PLOT if 0 <= q < T]
    x = np.arange(T, dtype=np.float64)
    normalized = {}
    for model_info in MODELS:
        scale = float(model_info["rho_scale"])
        u = x / scale
        wave = (1.0 - u**2) * np.exp(-0.5 * u**2)
        normalized[scale] = wave / max(_rms(wave), 1e-12)

    fig, axes = plt.subplots(len(query_rows), 2, figsize=(17, 4.0 * len(query_rows)), squeeze=False)
    colors = ["#2a9d8f", "#e9c46a", "#e76f51"]
    for row_idx, q in enumerate(query_rows):
        keys = np.arange(q + 1)
        for color, model_info in zip(colors, MODELS):
            scale = float(model_info["rho_scale"])
            visible = normalized[scale][: q + 1]
            centered = visible - float(np.mean(visible))
            axes[row_idx, 0].plot(
                keys,
                visible,
                color=color,
                linewidth=1.3,
                label=f"rho={scale:.0f} ({model_info['fraction']:.2f}L)",
            )
            axes[row_idx, 1].plot(
                keys,
                centered,
                color=color,
                linewidth=1.3,
                label=(
                    f"rho={scale:.0f}: centered/raw RMS="
                    f"{_rms(centered) / max(_rms(visible), 1e-12):.2f}"
                ),
            )
        axes[row_idx, 0].axhline(0.0, color="0.4", linewidth=0.5)
        axes[row_idx, 1].axhline(0.0, color="0.4", linewidth=0.5)
        axes[row_idx, 0].set_title(f"q={q}: context-RMS-normalized Ricker, g0=1, shift=0")
        axes[row_idx, 1].set_title(f"q={q}: row-centered component that can affect softmax")
        for col in range(2):
            axes[row_idx, col].set_xlabel("visible key position k")
            axes[row_idx, col].set_ylabel("wavelet logit bias")
            axes[row_idx, col].legend(loc="best", fontsize=8)

    fig.suptitle(
        "Controlled K=1 time-domain comparison: isolate Ricker scale from learned g0 and shift\n"
        "Zero crossing occurs at k=rho; subtracting each visible row mean removes the softmax-invariant component",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = OUT_DIR / f"controlled_k1_ricker_time_domain_L{T}.png"
    fig.savefig(out_path, dpi=145)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    batch = build_batch(tokenizer, T, n=1).to(device)

    print("Loading trained PA-only baseline...")
    pa_model = load_model(PA_ONLY_CHECKPOINT, PA_ONLY_CFG_PATH, T)
    pa_model.eval().to(device)
    pa_baseline, _, _ = run_and_capture(pa_model, batch, LAYERS_TO_PLOT)
    del pa_model
    torch.cuda.empty_cache()

    results = {}
    for model_info in MODELS:
        checkpoint, cfg_path = _model_paths(model_info["run"])
        print(f"Loading {model_info['run']}...")
        model = load_model(checkpoint, cfg_path, T)
        model.eval().to(device)
        pa_component, full_component, diagnostics = run_and_capture(
            model, batch, LAYERS_TO_PLOT, capture_full=True
        )
        results[model_info["tag"]] = {
            "pa_baseline": pa_baseline,
            "pa_component": pa_component,
            "full_component": full_component,
            "diagnostics": diagnostics,
        }
        del model
        torch.cuda.empty_cache()
        plot_heatmap(model_info, pa_baseline, pa_component, full_component)

    for layer in LAYERS_TO_PLOT:
        plot_row_sweep(layer, results)
    write_metrics(results)
    plot_controlled_time_domain()


if __name__ == "__main__":
    main()

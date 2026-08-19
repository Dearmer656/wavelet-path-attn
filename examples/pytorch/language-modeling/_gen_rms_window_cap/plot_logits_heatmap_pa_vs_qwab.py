#!/usr/bin/env python3
"""L_test x L_test raw (unnormalized, pre-softmax) attention-logit heatmap, 4-way:

  1) Trained-PA-only baseline (token_even_mix_PA_s42, wavelet never active during training)
  2) QWAB checkpoint's own PA component (E_base_raw, extracted mid-forward via the
     _last_logits_pa_only hook) -- same weights that also produced (3)
  3) QWAB checkpoint's full output (PA + wavelet bias, _last_logits_full)
  4) diff = (2) - (1): does training WITH QWAB active reshape what PA itself learns,
     independent of the wavelet bias's own additive contribution (which is (3)-(2))?

Different checkpoints in (1) vs (2)/(3) -- both trained from the same
1r_baseline_from_s/checkpoint-80000 start on the same token_even_mix data, same steps,
only difference is whether the wavelet mechanism was active during training.
"""
import sys
import importlib
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling")
import run_clm  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

QWAB_CHECKPOINT = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat244_dual_temp/K3_L512_fixedratioRmsBoth_128_256_384_s42/checkpoint-15000"
QWAB_CFG_PATH = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat244_dual_temp/K3_L512_fixedratioRmsBoth_128_256_384_s42/supply_model.cfg"
PA_ONLY_CHECKPOINT = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/PA_baseline_multi_seeds/token_even_mix_PA_s42/checkpoint-15000"
PA_ONLY_CFG_PATH = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/PA_baseline_multi_seeds/token_even_mix_PA_s42/supply_model.cfg"
T = 2048
LAYERS_TO_PLOT = [0, 2, 6, 8]  # 0 = generic; 2/6 = the high-shift layers found earlier; 8 = notable p90 drop layer
QUERY_ROWS_TO_PLOT = [255, 511, 1023, 2047]
OUT_PATH = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_rms_window_cap/logits_heatmap_pa_vs_qwab_L2048.png"
ROW_OUT_DIR = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_rms_window_cap"
METRICS_PATH = f"{ROW_OUT_DIR}/logits_row_diagnostics_L2048.csv"


def load_model(checkpoint_dir, cfg_path, block_size):
    config = AutoConfig.from_pretrained(checkpoint_dir)
    cfg = run_clm.read_kv_config(str(cfg_path))
    run_clm.add_missing_to_hf_config(config, cfg)
    wavelet_mode = str(getattr(config, "wavelet_mode", "")).strip().lower()
    bias_type = str(getattr(config, "bias_type", "")).strip().lower()
    ctxscale_active = bias_type and wavelet_mode in {
        "logit_bias_ctxscale_shift_v0",
        "logit_bias_ctxscale_shift_v0_film",
        "mlp_bias_baseline_v0",
    }
    if not ctxscale_active:
        # Old PA-only checkpoints predate the K>1 scale-list validation. Keep
        # their K=8 module shapes for state-dict compatibility, while expanding
        # the old [2^0, ..., 2^max_exp] grid into the current per-scale format.
        k = int(getattr(config, "wavelet_ctxscale_k", 8))
        max_exp = getattr(config, "wavelet_ctxscale_scale_max_exp", 14.0)
        config.wavelet_ctxscale_k = k
        if k > 1 and not isinstance(max_exp, (list, tuple)):
            max_exp = float(max_exp)
            config.wavelet_ctxscale_scale_max_exp = [
                2.0 * max_exp * i / (k - 1) for i in range(k)
            ]
    config.attn_implementation = "path_attn"
    config.use_cache = False
    config.block_size = block_size
    return AutoModelForCausalLM.from_pretrained(checkpoint_dir, config=config)


def build_batch(tokenizer, length, n=1, seed_tokens=None):
    if seed_tokens is not None:
        return seed_tokens
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    text_parts = [row["text"] for row in dataset if row.get("text", "").strip()]
    corpus = "\n\n".join(text_parts)
    token_ids = tokenizer(corpus, add_special_tokens=False)["input_ids"]
    needed = n * length
    token_ids = token_ids[:needed]
    return torch.tensor(token_ids, dtype=torch.long).view(n, length)


def _head_mean_to_numpy(logits):
    return logits.detach()[0].mean(dim=0).float().cpu().numpy()


def run_and_capture(model, input_ids, layers, capture_full=False):
    """Run one forward and capture raw PA logits without changing its outputs."""
    cores = {
        layer: getattr(model.transformer.h[layer].attn, "core", model.transformer.h[layer].attn)
        for layer in layers
    }
    path_module = importlib.import_module(next(iter(cores.values())).__class__.__module__)
    original_path_ut_base_raw = path_module.path_ut_base_raw
    active_layer = {"value": None}
    pa_logits = {}
    full_logits = {}
    diagnostics = {}
    handles = []

    def capture_path_ut_base_raw(*args, **kwargs):
        result = original_path_ut_base_raw(*args, **kwargs)
        layer = active_layer["value"]
        if layer in cores:
            pa_logits[layer] = _head_mean_to_numpy(result[0])
        return result

    def make_pre_hook(layer):
        def pre_hook(module, args):
            active_layer["value"] = layer
            for key in ("_last_logits_pa_only", "_last_logits_full"):
                if hasattr(module, key):
                    delattr(module, key)

        return pre_hook

    def make_post_hook(layer):
        def post_hook(module, args, output):
            if capture_full:
                value = getattr(module, "_last_logits_full", None)
                if value is not None:
                    full_logits[layer] = _head_mean_to_numpy(value)
                diagnostics[layer] = {}
                attr_names = {
                    "g0": "_last_ctxscale_non_null_mass",
                    "rho": "_last_ctxscale_rho",
                    "beta": "_last_ctxscale_beta_m",
                    "scale_prob": "_last_ctxscale_router_prob",
                }
                for name, attr_name in attr_names.items():
                    attr_value = getattr(module, attr_name, None)
                    if torch.is_tensor(attr_value):
                        diagnostics[layer][name] = attr_value.detach().float().cpu().numpy()
                scales = getattr(module, "wavelet_ctxscale_scales", None)
                if torch.is_tensor(scales):
                    diagnostics[layer]["scales"] = scales.detach().float().cpu().numpy()
            for key in ("_last_logits_pa_only", "_last_logits_full"):
                if hasattr(module, key):
                    delattr(module, key)
            active_layer["value"] = None

        return post_hook

    path_module.path_ut_base_raw = capture_path_ut_base_raw
    try:
        for layer, core in cores.items():
            handles.append(core.register_forward_pre_hook(make_pre_hook(layer)))
            handles.append(core.register_forward_hook(make_post_hook(layer)))
        with torch.inference_mode():
            model(input_ids=input_ids)
    finally:
        path_module.path_ut_base_raw = original_path_ut_base_raw
        for handle in handles:
            handle.remove()

    missing_pa = sorted(set(layers) - set(pa_logits))
    missing_full = sorted(set(layers) - set(full_logits)) if capture_full else []
    if missing_pa or missing_full:
        raise RuntimeError(
            f"failed to capture logits: missing PA layers={missing_pa}, "
            f"missing full layers={missing_full}"
        )
    return pa_logits, full_logits, diagnostics


def _rms(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else float("nan")


def _corr(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _vector_from_diag(value, length):
    value = np.asarray(value)
    value = np.squeeze(value)
    if value.ndim == 1 and value.shape[0] == length:
        return value
    if value.ndim >= 2 and value.shape[0] == length:
        return value.mean(axis=tuple(range(1, value.ndim)))
    raise ValueError(f"cannot reduce diagnostic shape {value.shape} to length {length}")


def write_metrics_and_row_plots(pa_baseline, pa_component, full_component, diagnostics):
    rows = []
    query_rows = [q for q in QUERY_ROWS_TO_PLOT if 0 <= q < T]
    for layer in LAYERS_TO_PLOT:
        pa_base = pa_baseline[layer]
        pa_comp = pa_component[layer]
        full = full_component[layer]
        wavelet_bias = full - pa_comp
        training_imprint = pa_comp - pa_base
        total_difference = full - pa_base
        diag = diagnostics[layer]
        g0 = _vector_from_diag(diag["g0"], T)
        rho = _vector_from_diag(diag["rho"], T)
        beta = _vector_from_diag(diag["beta"], T)
        scales = np.asarray(diag["scales"]).reshape(-1)

        causal_mask = np.tri(T, T, dtype=bool)
        causal_bias_sq = np.where(causal_mask, wavelet_bias**2, 0.0).sum(axis=1)
        causal_bias_rms = np.sqrt(causal_bias_sq / np.arange(1, T + 1))
        causal_total_sq = np.where(causal_mask, total_difference**2, 0.0).sum(axis=1)
        causal_total_rms = np.sqrt(causal_total_sq / np.arange(1, T + 1))
        full_bias_rms = np.sqrt(np.mean(wavelet_bias**2, axis=1))

        visible_train = training_imprint[causal_mask]
        visible_bias = wavelet_bias[causal_mask]
        visible_total = total_difference[causal_mask]
        visible_pa = pa_comp[causal_mask]
        rows.append({
            "scope": "layer_causal_all",
            "layer": layer,
            "query": "all",
            "g0": float(np.mean(g0)),
            "rho": float(np.mean(rho)),
            "beta_unit": float(np.mean(beta)),
            "pa_rms": _rms(visible_pa),
            "training_imprint_rms": _rms(visible_train),
            "wavelet_bias_rms": _rms(visible_bias),
            "full_vs_baseline_rms": _rms(visible_total),
            "full_vs_baseline_over_pa_rms": _rms(visible_total) / max(_rms(visible_pa), 1e-12),
            "wavelet_over_pa_rms": _rms(visible_bias) / max(_rms(visible_pa), 1e-12),
            "wavelet_over_training_imprint_rms": _rms(visible_bias) / max(_rms(visible_train), 1e-12),
            "corr_wavelet_pa": _corr(visible_bias, visible_pa),
            "corr_wavelet_training_imprint": _corr(visible_bias, visible_train),
            "corr_g0_full_context_bias_rms": _corr(g0, full_bias_rms),
            "corr_g0_visible_bias_rms": _corr(g0, causal_bias_rms),
            "corr_g0_visible_full_vs_baseline_rms": _corr(g0, causal_total_rms),
            "corr_wavelet_full_vs_baseline": _corr(visible_bias, visible_total),
            "mean_abs_full_context_bias_rms_minus_g0": float(np.mean(np.abs(full_bias_rms - g0))),
        })

        fig, axes = plt.subplots(len(query_rows) + 1, 3, figsize=(22, 3.8 * (len(query_rows) + 1)))
        q_axis = np.arange(T)
        axes[0, 0].plot(q_axis, g0, color="#087e8b", linewidth=1.2)
        axes[0, 0].set_title(f"layer {layer}: non-null gate $g_0(q)$")
        axes[0, 0].set_ylim(-0.02, 1.02)
        axes[0, 0].set_xlabel("query position q")
        axes[0, 0].set_ylabel("g0")

        axes[0, 1].plot(q_axis, rho, color="#d1495b", linewidth=1.1, label="rho")
        axes[0, 1].plot(q_axis, beta, color="#edae49", linewidth=1.0, label="beta unit")
        axes[0, 1].set_title("query-conditioned shift controls")
        axes[0, 1].set_xlabel("query position q")
        axes[0, 1].legend(loc="best")

        axes[0, 2].plot(q_axis, full_bias_rms, color="#5f0f40", linewidth=1.1, label="bias RMS, full context")
        axes[0, 2].plot(q_axis, causal_bias_rms, color="#fb8b24", linewidth=1.0, label="bias RMS, visible k<=q")
        axes[0, 2].plot(q_axis, g0, color="#0f4c5c", linewidth=0.9, alpha=0.8, label="g0")
        axes[0, 2].set_title("does row-wise QWAB amplitude track g0?")
        axes[0, 2].set_xlabel("query position q")
        axes[0, 2].legend(loc="best")

        for q in query_rows:
            for ax in axes[0]:
                ax.axvline(q, color="0.45", linewidth=0.6, alpha=0.45)
            row_idx = query_rows.index(q) + 1
            keys = np.arange(q + 1)
            base_row = pa_base[q, : q + 1]
            pa_row = pa_comp[q, : q + 1]
            full_row = full[q, : q + 1]
            bias_row = wavelet_bias[q, : q + 1]
            imprint_row = training_imprint[q, : q + 1]
            g0_q = float(g0[q])
            conditional_pattern = bias_row / max(abs(g0_q), 1e-8)
            bias_rms = _rms(bias_row)
            pa_rms = _rms(pa_row)
            imprint_rms = _rms(imprint_row)
            total_rms = _rms(full_row - base_row)

            axes[row_idx, 0].plot(keys, base_row, color="0.55", linewidth=0.75, label="trained PA-only")
            axes[row_idx, 0].plot(keys, pa_row, color="#1d3557", linewidth=0.8, label="QWAB ckpt PA")
            axes[row_idx, 0].plot(keys, full_row, color="#e63946", linewidth=0.8, alpha=0.85, label="QWAB full")
            axes[row_idx, 0].set_title(f"q={q}: raw mean-head logits")
            axes[row_idx, 0].set_ylabel("logit")
            axes[row_idx, 0].legend(loc="best", fontsize=7)

            axes[row_idx, 1].plot(keys, conditional_pattern, color="#457b9d", linewidth=1.0, label="bias_eff / g0")
            axes[row_idx, 1].plot(keys, bias_row, color="#e63946", linewidth=1.0, label="bias_eff = full - PA")
            axes[row_idx, 1].axhline(0.0, color="0.4", linewidth=0.5)
            axes[row_idx, 1].set_title(
                f"wavelet row: g0={g0_q:.3f}, rho={rho[q]:.3f}, beta_unit={beta[q]:.3f}"
            )
            axes[row_idx, 1].set_ylabel("wavelet logit bias")
            axes[row_idx, 1].legend(loc="best", fontsize=7)

            axes[row_idx, 2].plot(keys, imprint_row, color="#6a4c93", linewidth=0.85, label="training imprint: QWAB_PA - PA-only")
            axes[row_idx, 2].plot(keys, bias_row, color="#f77f00", linewidth=0.95, label="direct wavelet bias")
            axes[row_idx, 2].axhline(0.0, color="0.4", linewidth=0.5)
            axes[row_idx, 2].set_title(
                f"two QWAB effects: RMS imprint={imprint_rms:.3f}, bias={bias_rms:.3f}, PA={pa_rms:.3f}"
            )
            axes[row_idx, 2].set_ylabel("delta logit")
            axes[row_idx, 2].legend(loc="best", fontsize=7)

            for col in range(3):
                axes[row_idx, col].set_xlabel("key position k (visible keys only)")

            rows.append({
                "scope": "query_row_causal",
                "layer": layer,
                "query": q,
                "g0": g0_q,
                "rho": float(rho[q]),
                "beta_unit": float(beta[q]),
                "pa_rms": pa_rms,
                "training_imprint_rms": imprint_rms,
                "wavelet_bias_rms": bias_rms,
                "full_vs_baseline_rms": total_rms,
                "full_vs_baseline_over_pa_rms": total_rms / max(pa_rms, 1e-12),
                "wavelet_over_pa_rms": bias_rms / max(pa_rms, 1e-12),
                "wavelet_over_training_imprint_rms": bias_rms / max(imprint_rms, 1e-12),
                "corr_wavelet_pa": _corr(bias_row, pa_row),
                "corr_wavelet_training_imprint": _corr(bias_row, imprint_row),
                "corr_g0_full_context_bias_rms": "",
                "corr_g0_visible_bias_rms": "",
                "corr_g0_visible_full_vs_baseline_rms": "",
                "corr_wavelet_full_vs_baseline": _corr(bias_row, full_row - base_row),
                "mean_abs_full_context_bias_rms_minus_g0": "",
            })

        fig.suptitle(
            f"Layer {layer} row-level QWAB diagnostics at L={T}; scales={scales.tolist()}\n"
            "Exact identity: QWAB full logits = QWAB PA logits + effective wavelet bias",
            fontsize=13,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        row_path = f"{ROW_OUT_DIR}/logits_row_diagnostics_layer{layer}_L{T}.png"
        fig.savefig(row_path, dpi=140)
        plt.close(fig)
        print(f"Wrote {row_path}")

    with open(METRICS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {METRICS_PATH}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    batch = build_batch(tokenizer, T, n=1)

    print("Loading QWAB checkpoint...")
    qwab_model = load_model(QWAB_CHECKPOINT, QWAB_CFG_PATH, T)
    qwab_model.eval().to(device)
    pa_component, full_component, qwab_diagnostics = run_and_capture(
        qwab_model, batch.to(device), LAYERS_TO_PLOT, capture_full=True
    )
    del qwab_model
    torch.cuda.empty_cache()

    print("Loading trained-PA-only baseline checkpoint...")
    pa_model = load_model(PA_ONLY_CHECKPOINT, PA_ONLY_CFG_PATH, T)
    pa_model.eval().to(device)
    pa_baseline, _, _ = run_and_capture(pa_model, batch.to(device), LAYERS_TO_PLOT)
    del pa_model
    torch.cuda.empty_cache()

    n_layers = len(LAYERS_TO_PLOT)
    fig, axes = plt.subplots(n_layers, 4, figsize=(21, 4.5 * n_layers))
    if n_layers == 1:
        axes = axes.reshape(1, -1)

    for row, layer in enumerate(LAYERS_TO_PLOT):
        pa_base = pa_baseline.get(layer)
        pa_comp = pa_component.get(layer)
        full = full_component.get(layer)
        if pa_base is None or pa_comp is None or full is None:
            print(f"layer {layer}: MISSING DATA (pa_base={pa_base is not None}, pa_comp={pa_comp is not None}, full={full is not None})")
            continue

        causal_mask = np.triu(np.ones_like(pa_base, dtype=bool), k=1)
        pa_base_disp = np.where(causal_mask, np.nan, pa_base)
        pa_comp_disp = np.where(causal_mask, np.nan, pa_comp)
        full_disp = np.where(causal_mask, np.nan, full)
        train_diff = pa_comp - pa_base  # did training WITH QWAB reshape PA itself?
        train_diff_disp = np.where(causal_mask, np.nan, train_diff)

        vmin = np.nanmin([pa_base_disp, pa_comp_disp, full_disp])
        vmax = np.nanmax([pa_base_disp, pa_comp_disp, full_disp])

        im0 = axes[row, 0].imshow(pa_base_disp, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
        axes[row, 0].set_title(f"layer {layer}: trained PA-only baseline")
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046)

        im1 = axes[row, 1].imshow(pa_comp_disp, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
        axes[row, 1].set_title(f"layer {layer}: QWAB ckpt's PA component")
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046)

        im2 = axes[row, 2].imshow(full_disp, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
        axes[row, 2].set_title(f"layer {layer}: QWAB ckpt full (PA+wavelet)")
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046)

        dmax = np.nanmax(np.abs(train_diff_disp))
        im3 = axes[row, 3].imshow(train_diff_disp, cmap="PuOr", vmin=-dmax, vmax=dmax, aspect="auto")
        axes[row, 3].set_title(f"layer {layer}: diff = QWAB's-PA − trained-PA-only\n(training-stage imprint)")
        plt.colorbar(im3, ax=axes[row, 3], fraction=0.046)

        for ax in axes[row]:
            ax.set_xlabel("key position")
            ax.set_ylabel("query position")

    fig.suptitle(
        f"Raw (unnormalized, pre-softmax) attention logits at L_test={T}\n"
        f"PA-only baseline: token_even_mix_PA_s42  |  QWAB: fixedratioRmsBoth_s42  |  mean over heads, causal",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_PATH, dpi=130)
    print(f"Wrote {OUT_PATH}")
    write_metrics_and_row_plots(pa_baseline, pa_component, full_component, qwab_diagnostics)


if __name__ == "__main__":
    main()

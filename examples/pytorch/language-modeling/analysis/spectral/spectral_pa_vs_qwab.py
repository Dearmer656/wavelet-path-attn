#!/usr/bin/env python3
"""Length-extrapolation diagnostic: PA-only vs QWAB attention-logit spectra.

This script measures the attention-logit frequency spectrum for a fixed query
near the end of the sequence, compares PA-only against QWAB at L=512/2048/4096,
and saves both a PDF figure and raw JSON payload for later reuse.
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from types import MethodType

import numpy as np
import torch

sys.path.insert(0, "/project/nlp-work5/hongyu-s/transformers/src")
sys.path.insert(0, "/project/nlp-work5/hongyu-s/flash-linear-attention")
sys.path.insert(0, "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling")

import fla.models  # noqa: F401,E402
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402
import run_clm  # noqa: E402


WORKDIR = Path("/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling")
OUT_DIR = WORKDIR / "analysis" / "paper_figures"
JSON_OUT = OUT_DIR / "spectral_pa_only_vs_qwab_by_length.json"
PDF_OUT = OUT_DIR / "spectral_pa_only_vs_qwab_by_length.pdf"

PA_CKPT = WORKDIR / "runs/PA_baseline_multi_seeds/token_even_mix_PA_s42/checkpoint-15000"
PA_CFG = WORKDIR / "_gen_headline_extrap_curve/PAonly_s42_fixed.cfg"
QWAB_CKPT = WORKDIR / "runs/pat244_dual_temp/K1_L512_me16_rho256_ricker_s42/checkpoint-15000"
QWAB_CFG = QWAB_CKPT.parent / "supply_model.cfg"

WIKITEXT_CANDIDATES = [
    WORKDIR / "wikitext-103-raw-v1" / "wiki.valid.raw",
    WORKDIR / "wikitext-103-raw-v1" / "valid.txt",
    WORKDIR / "wikitext-103-raw-v1" / "validation.txt",
    WORKDIR / "data" / "wikitext-103-raw-v1" / "wiki.valid.raw",
    WORKDIR / "data" / "wikitext-103-raw-v1" / "valid.txt",
]
HOTPOT_JSONL = WORKDIR / "hotpot_long" / "data" / "hotpot_long_dev_uniform.jsonl"

LENGTHS = [512, 2048, 4096]
N_EXAMPLES = 8
QUERY_INDEX_FROM_END = 1
FFT_WINDOW = "hann"
FFT_STRIP_DC = True
FFT_AVERAGE_OVER = ("batch", "heads")
N_LAYERS = 12


def _load_texts() -> tuple[list[str], str]:
    for path in WIKITEXT_CANDIDATES:
        if path.exists():
            texts = path.read_text().splitlines()
            texts = [t.strip() for t in texts if t.strip()]
            if texts:
                return texts, f"wikitext:{path}"

    texts: list[str] = []
    with HOTPOT_JSONL.open() as f:
        for line in f:
            rec = json.loads(line)
            context = rec.get("context", [])
            if isinstance(context, list):
                ctx = " ".join(
                    f"{title}: {' '.join(sentences) if isinstance(sentences, list) else str(sentences)}"
                    for title, sentences in context
                )
            else:
                ctx = str(context)
            question = rec.get("question", "")
            answer = rec.get("answer", "")
            texts.append(f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer: {answer}")
            if len(texts) >= 200:
                break
    return texts, f"hotpot:{HOTPOT_JSONL}"


def _chunk_token_ids(tokenizer, texts: list[str], seq_len: int, n_examples: int) -> torch.Tensor:
    all_ids: list[int] = []
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        all_ids.extend(ids)
        if len(all_ids) >= seq_len * n_examples:
            break
    if len(all_ids) < seq_len:
        raise ValueError("Not enough tokens to build one evaluation block.")
    n_blocks = min(n_examples, len(all_ids) // seq_len)
    blocks = [all_ids[i * seq_len : (i + 1) * seq_len] for i in range(n_blocks)]
    if len(blocks) < n_examples:
        raise ValueError(f"Only built {len(blocks)} blocks; need {n_examples}.")
    return torch.tensor(blocks[:n_examples], dtype=torch.long)


def _load_eval_batch(tokenizer, seq_len: int) -> tuple[torch.Tensor, str]:
    texts, source = _load_texts()
    return _chunk_token_ids(tokenizer, texts, seq_len, N_EXAMPLES), source


def _load_model(checkpoint_dir: Path, cfg_path: Path) -> torch.nn.Module:
    config = AutoConfig.from_pretrained(checkpoint_dir)
    cfg = run_clm.read_kv_config(str(cfg_path))
    run_clm.add_missing_to_hf_config(config, cfg)
    config.attn_implementation = "path_attn"
    config.use_cache = False
    return AutoModelForCausalLM.from_pretrained(checkpoint_dir, config=config)


@contextlib.contextmanager
def _capture_qwab_logits(model):
    from fla.layers import path_attn as path_attn_mod

    captured: dict[int, torch.Tensor] = {}
    original = path_attn_mod.PaTHAttention._build_ctxscale_shift_logit_bias_v0

    def wrapped(self, *args, **kwargs):
        logits_out, payload = original(self, *args, **kwargs)
        layer_idx = int(getattr(self, "layer_idx", -1))
        captured[layer_idx] = logits_out.detach().float().cpu()
        return logits_out, payload

    path_attn_mod.PaTHAttention._build_ctxscale_shift_logit_bias_v0 = wrapped
    try:
        yield captured
    finally:
        path_attn_mod.PaTHAttention._build_ctxscale_shift_logit_bias_v0 = original


@contextlib.contextmanager
def _capture_pa_raw_logits(model):
    import transformers.models.gpt2.modeling_gpt2 as modeling_gpt2
    from fla.layers import path_attn as path_attn_mod

    captured: dict[int, torch.Tensor] = {}
    original_module = path_attn_mod.path_ut_base_raw
    original_alias = modeling_gpt2._path_ut_base_raw
    original_forward = modeling_gpt2.GPT2PaTHAttention.forward
    current_layer = {"idx": None}

    def wrapped(*args, **kwargs):
        out = original_module(*args, **kwargs)
        if current_layer["idx"] is not None:
            captured[int(current_layer["idx"])] = out[0].detach().float().cpu()
        return out

    def wrapped_forward(self, *args, **kwargs):
        current_layer["idx"] = getattr(self, "layer_idx", None)
        try:
            return original_forward(self, *args, **kwargs)
        finally:
            current_layer["idx"] = None

    path_attn_mod.path_ut_base_raw = wrapped
    modeling_gpt2._path_ut_base_raw = wrapped
    modeling_gpt2.GPT2PaTHAttention.forward = wrapped_forward
    try:
        yield captured
    finally:
        path_attn_mod.path_ut_base_raw = original_module
        modeling_gpt2._path_ut_base_raw = original_alias
        modeling_gpt2.GPT2PaTHAttention.forward = original_forward


def _layer_query_row_spectrum(logits_bhtt: torch.Tensor, query_index: int) -> np.ndarray:
    logits = logits_bhtt.mean(dim=(0, 1)).detach().float().cpu().numpy()  # [T, T]
    signal = logits[query_index]
    signal = signal - signal.mean()
    if FFT_WINDOW == "hann":
        signal = signal * np.hanning(signal.shape[-1])
    spectrum = np.abs(np.fft.rfft(signal))
    return spectrum.astype(np.float64)


def _collect_model_spectra(model, batch: torch.Tensor, model_name: str) -> dict:
    captured: dict[int, torch.Tensor] = {}
    if model_name == "qwab":
        ctx = _capture_qwab_logits(model)
    elif model_name == "pa_only":
        ctx = _capture_pa_raw_logits(model)
    else:
        raise ValueError(model_name)

    with ctx as captured_logits:
        with torch.no_grad():
            model(input_ids=batch)
        captured = dict(captured_logits)

    spectra_by_length = []
    for layer_idx in range(N_LAYERS):
        logits = captured[layer_idx]
        q_idx = logits.shape[-1] - QUERY_INDEX_FROM_END
        spectra_by_length.append(_layer_query_row_spectrum(logits, q_idx))

    spectra = np.stack(spectra_by_length, axis=0)
    mean_spectrum = spectra.mean(axis=0)
    n_freq = mean_spectrum.shape[0]
    freqs = np.fft.rfftfreq(batch.shape[1], d=1.0).astype(np.float64)
    return {
        "layer_spectra": spectra.tolist(),
        "mean_spectrum": mean_spectrum.tolist(),
        "frequencies": freqs.tolist(),
        "num_freq_bins": int(n_freq),
    }


def _peak_excluding_dc(freqs: np.ndarray, spectrum: np.ndarray) -> tuple[int, float, float]:
    if spectrum.shape[0] < 2:
        return 0, float(freqs[0]), float(spectrum[0])
    idx = int(np.argmax(spectrum[1:]) + 1)
    return idx, float(freqs[idx]), float(spectrum[idx])


def _high_freq_ratio(pa_spectrum: np.ndarray, qwab_spectrum: np.ndarray) -> float:
    start = max(1, int(np.floor(0.75 * len(pa_spectrum))))
    pa = float(pa_spectrum[start:].sum())
    qwab = float(qwab_spectrum[start:].sum())
    return qwab / pa if pa > 0 else float("inf")


def _make_plot(payload: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    colors = {"pa_only": "#1f77b4", "qwab": "#d62728"}
    labels = {"pa_only": "PA-only", "qwab": "QWAB"}
    y_max = 0.0
    y_min = None
    for length_key in [str(L) for L in LENGTHS]:
        for model_name in ("pa_only", "qwab"):
            arr = np.asarray(payload["results"][length_key][model_name]["mean_spectrum"], dtype=np.float64)
            y_max = max(y_max, float(arr.max()))
            positive = arr[arr > 0]
            if positive.size:
                cand = float(positive.min())
                y_min = cand if y_min is None else min(y_min, cand)
    y_min = max(y_min or 1e-12, 1e-12)

    for ax, length in zip(axes, LENGTHS):
        entry = payload["results"][str(length)]
        for model_name in ("pa_only", "qwab"):
            freqs = np.asarray(entry[model_name]["frequencies"], dtype=np.float64)
            spectrum = np.asarray(entry[model_name]["mean_spectrum"], dtype=np.float64)
            ax.plot(freqs, spectrum, color=colors[model_name], label=labels[model_name], lw=1.8)
        ax.set_title(f"L={length}")
        ax.set_xlabel("Frequency (cycles/token)")
        ax.set_yscale("log")
        ax.set_xlim(0.0, 0.5)
        ax.set_ylim(y_min, y_max * 1.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel("Magnitude")
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("PA-only vs QWAB (Ricker, rho=256): attention-logit spectrum vs sequence length")
    fig.tight_layout()
    fig.savefig(PDF_OUT, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pa_model = _load_model(PA_CKPT, PA_CFG).eval().to(device)
    qwab_model = _load_model(QWAB_CKPT, QWAB_CFG).eval().to(device)

    results: dict[str, dict] = {}
    summary_lines: list[str] = []
    texts, source = _load_texts()

    for length in LENGTHS:
        batch = _chunk_token_ids(tokenizer, texts, length, N_EXAMPLES).to(device)
        pa = _collect_model_spectra(pa_model, batch, "pa_only")
        qwab = _collect_model_spectra(qwab_model, batch, "qwab")
        results[str(length)] = {"pa_only": pa, "qwab": qwab}

        pa_mean = np.asarray(pa["mean_spectrum"], dtype=np.float64)
        qwab_mean = np.asarray(qwab["mean_spectrum"], dtype=np.float64)
        freqs = np.asarray(pa["frequencies"], dtype=np.float64)
        pa_peak_idx, pa_peak_freq, pa_peak_mag = _peak_excluding_dc(freqs, pa_mean)
        qwab_peak_idx, qwab_peak_freq, qwab_peak_mag = _peak_excluding_dc(freqs, qwab_mean)
        hf_ratio = _high_freq_ratio(pa_mean, qwab_mean)
        summary_lines.append(
            f"L={length}: PA peak bin {pa_peak_idx} @ {pa_peak_freq:.6f} cyc/token -> {pa_peak_mag:.6e}; "
            f"QWAB peak bin {qwab_peak_idx} @ {qwab_peak_freq:.6f} cyc/token -> {qwab_peak_mag:.6e}; "
            f"high-freq ratio (QWAB/PA)={hf_ratio:.4f}"
        )

    payload = {
        "source": source,
        "lengths": LENGTHS,
        "n_examples": N_EXAMPLES,
        "query_index_from_end": QUERY_INDEX_FROM_END,
        "fft_window": FFT_WINDOW,
        "fft_strip_dc": FFT_STRIP_DC,
        "results": results,
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2))
    _make_plot(payload)

    print(f"source={source}")
    for line in summary_lines:
        print(line)


if __name__ == "__main__":
    main()

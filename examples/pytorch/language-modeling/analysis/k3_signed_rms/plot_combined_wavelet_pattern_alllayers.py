#!/usr/bin/env python3
"""Plot the actual combined wavelet pattern used at checkpoint-15000, per layer."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling")
import run_clm  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402


CHECKPOINT = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat244_dual_temp/K3_L512_signed_128_256_384_s42/checkpoint-15000"
CFG_PATH = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat244_dual_temp/K3_L512_signed_128_256_384_s42/supply_model.cfg"
OUT_PATH = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/analysis/k3_signed_rms/combined_wavelet_pattern_alllayers_ckpt15000.png"
N_LAYERS = 12
BLOCK_SIZE = 512
SCALES = np.array([128.0, 256.0, 384.0], dtype=np.float64)
DELTA = np.linspace(-600.0, 600.0, 1200, dtype=np.float64)
EPS = 1e-6


def load_model(checkpoint_dir: str, cfg_path: str):
    config = AutoConfig.from_pretrained(checkpoint_dir)
    cfg = run_clm.read_kv_config(str(cfg_path))
    run_clm.add_missing_to_hf_config(config, cfg)
    config.attn_implementation = "path_attn"
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, config=config)
    return model


def ricker(u: np.ndarray) -> np.ndarray:
    return (1.0 - u**2) * np.exp(-0.5 * u**2)


def combined_pattern_shifted(weights: np.ndarray, beta_m: float) -> np.ndarray:
    basis = np.stack([ricker(DELTA / scale - beta_m) for scale in SCALES], axis=0)
    return np.sum(weights[:, None] * basis, axis=0)


def rms_normalize(values: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.mean(values**2) + EPS)
    return values / scale


def build_input_batch(tokenizer: AutoTokenizer) -> torch.Tensor:
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    text_parts = [row["text"] for row in dataset if row.get("text", "").strip()]
    corpus = "\n\n".join(text_parts)
    token_ids = tokenizer(corpus, add_special_tokens=False)["input_ids"]
    needed = 4 * BLOCK_SIZE
    token_ids = token_ids[:needed]
    if len(token_ids) < needed:
        raise RuntimeError(f"Need at least {needed} tokens, got {len(token_ids)}")
    batch = torch.tensor(token_ids, dtype=torch.long).view(4, BLOCK_SIZE)
    return batch


def extract_layer_stats(model: torch.nn.Module, layer: int):
    core = getattr(model.transformer.h[layer].attn, "core", model.transformer.h[layer].attn)
    router_prob = getattr(core, "_last_ctxscale_router_prob", None)
    beta_m = getattr(core, "_last_ctxscale_beta_m", None)
    if router_prob is None or beta_m is None:
        raise RuntimeError(f"Missing debug hooks on layer {layer}")
    router_prob = router_prob.detach().float().cpu()
    if router_prob.dim() == 4:
        router_prob = router_prob[:, :, 0, :]
    weights = router_prob.numpy().reshape(-1, router_prob.shape[-1])
    beta_m = beta_m.detach().float().cpu()
    if beta_m.dim() == 3:
        beta_m = beta_m[:, :, 0]
    beta_m = beta_m.numpy().reshape(-1)
    return weights, beta_m


def main() -> None:
    out_path = Path(OUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    batch = build_input_batch(tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(CHECKPOINT, CFG_PATH)
    model.eval().to(device)

    with torch.no_grad():
        model(input_ids=batch.to(device))

    basis = np.stack([ricker(DELTA / scale) for scale in SCALES], axis=0)

    fig, axes = plt.subplots(3, 4, figsize=(22, 14), sharex=True)
    axes = axes.ravel()

    print(f"{'layer':>5} {'w_128':>8} {'w_256':>8} {'w_384':>8} {'mean|beta_m|':>12}")
    for layer in range(N_LAYERS):
        weights, beta_m_all = extract_layer_stats(model, layer)
        mean_weights = weights.mean(axis=0)
        mean_beta_m = float(beta_m_all.mean())
        print(f"{layer:>5} {mean_weights[0]:>8.3f} {mean_weights[1]:>8.3f} {mean_weights[2]:>8.3f} {abs(mean_beta_m):>12.3f}")

        unshifted = rms_normalize(np.sum(mean_weights[:, None] * basis, axis=0))
        shifted = rms_normalize(combined_pattern_shifted(mean_weights, mean_beta_m))

        ax = axes[layer]
        ax.plot(DELTA, unshifted, color="tab:gray", linewidth=1.3, linestyle="--", label="beta=0")
        ax.plot(DELTA, shifted, color="tab:red", linewidth=1.8, label=f"beta_m={mean_beta_m:.2f}")
        ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.35)
        ax.set_title(
            f"layer {layer}: w=[{mean_weights[0]:.2f},{mean_weights[1]:.2f},{mean_weights[2]:.2f}]",
            fontsize=10,
        )
        ax.set_xlim(-600, 600)
        if layer == 0:
            ax.legend(fontsize=8)
        if layer >= 8:
            ax.set_xlabel("relative position δ (key − query)")
        if layer % 4 == 0:
            ax.set_ylabel("bias (RMS-normalized)")
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        "K3-signed s42, checkpoint-15000 (final): actual combined wavelet pattern by layer\n"
        "(mean per-scale weight over N=2048 query positions, real mean per-layer shift applied)"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

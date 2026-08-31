#!/usr/bin/env python3
"""Query/Key representation-similarity heatmaps: PaTH-only vs K1 rho=128 vs
K1 rho=256, small model, L4096.

For a sample of positions across the sequence, extracts the raw per-position
query and key vectors (post head-match, pre-PaTH-transform; concatenated
across heads -- this describes the model's own Q/K embedding geometry, not a
per-head attention pattern), and computes the full pairwise cosine
similarity matrix among sampled query vectors, and separately among sampled
key vectors, for each of the 3 models -- to see whether QWAB training
reshapes the Q/K representation geometry itself (not just the attention
bias) relative to a genuinely PA-only-trained backbone.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import sys

sys.path.insert(0, "/project/nlp-work5/hongyu-s/flash-linear-attention")
sys.path.insert(0, "/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long")

from fla.layers.path_attn import PaTHAttention  # noqa: F401
from dump_router_usage import render_prompt_hotpot, load_cases_hotpot, get_path_layers, load_path_attn_model
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def cosine_sim_matrix(vecs: torch.Tensor) -> np.ndarray:
    """vecs: [N, D] -> [N,N] cosine similarity."""
    v = vecs / (vecs.norm(dim=-1, keepdim=True) + 1e-12)
    return (v @ v.T).cpu().numpy()


def get_qk_at_layer(model, path_layers, tokenizer, rec, layer_idx, seq_len, device, sample_positions):
    prompt_ids = tokenizer(render_prompt_hotpot(rec), add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(f" {rec['answer']}", add_special_tokens=False)["input_ids"]
    full_ids = (prompt_ids + answer_ids)[:seq_len]
    if len(full_ids) < seq_len:
        raise RuntimeError(f"case too short: {len(full_ids)} < {seq_len}")
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        _ = model(input_ids)
    module = path_layers[layer_idx][1]
    q = module._last_q_vectors[0]  # [T,H,D]
    k = module._last_k_vectors[0]
    T, H, D = q.shape
    q_flat = q.reshape(T, H * D)
    k_flat = k.reshape(T, H * D)
    q_sample = q_flat[sample_positions]
    k_sample = k_flat[sample_positions]
    return q_sample, k_sample


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.pa_checkpoint, use_fast=False)

    models = {
        "PaTH-only": args.pa_checkpoint,
        "K1 rho=128": args.rho128_checkpoint,
        "K1 rho=256": args.rho256_checkpoint,
    }
    loaded = {}
    for tag, ckpt in models.items():
        print(f"Loading {tag}: {ckpt}")
        m = load_path_attn_model(ckpt, device, dtype=torch.float32)
        layers = get_path_layers(m)
        for _, mod in layers:
            mod._capture_debug_tensors = True
        loaded[tag] = (m, layers)

    cases = load_cases_hotpot(args.jsonl, args.seq_len, args.n_case)
    rec = cases[args.case_idx]

    n_pos = args.n_positions
    sample_positions = np.linspace(0, args.seq_len - 1, n_pos).astype(int).tolist()
    print(f"layer={args.layer} n_positions={n_pos} sample_positions[:5]={sample_positions[:5]}...")

    q_sims, k_sims = {}, {}
    for tag, (m, layers) in loaded.items():
        q_sample, k_sample = get_qk_at_layer(m, layers, tokenizer, rec, args.layer, args.seq_len, device, sample_positions)
        q_sims[tag] = cosine_sim_matrix(q_sample)
        k_sims[tag] = cosine_sim_matrix(k_sample)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    tags = list(models.keys())
    for ci, tag in enumerate(tags):
        im0 = axes[0][ci].imshow(q_sims[tag], cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        axes[0][ci].set_title(f"{tag}\nQuery-Query cosine sim", fontsize=10)
        im1 = axes[1][ci].imshow(k_sims[tag], cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        axes[1][ci].set_title(f"{tag}\nKey-Key cosine sim", fontsize=10)
        for ax in (axes[0][ci], axes[1][ci]):
            ax.set_xlabel("position idx (sampled)")
            ax.set_ylabel("position idx (sampled)")
    fig.colorbar(im0, ax=axes[0].tolist(), shrink=0.8, label="cosine sim")
    fig.colorbar(im1, ax=axes[1].tolist(), shrink=0.8, label="cosine sim")
    fig.suptitle(f"Query/Key representation similarity, layer={args.layer}, L={args.seq_len}, case={args.case_idx}, {n_pos} sampled positions", fontsize=13)
    out_path = OUT_DIR / f"qk_similarity_layer{args.layer}_case{args.case_idx}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pa_checkpoint", required=True)
    p.add_argument("--rho128_checkpoint", required=True)
    p.add_argument("--rho256_checkpoint", required=True)
    p.add_argument("--seq_len", type=int, default=4096)
    p.add_argument("--case_idx", type=int, default=0)
    p.add_argument("--n_case", type=int, default=5)
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--n_positions", type=int, default=64)
    p.add_argument("--jsonl", default="/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/hotpot_long_dev.jsonl")
    run(p.parse_args())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare raw query similarity across token positions for PaTH-only vs NoPE GPT-2-small.

The probe captures the raw query projection output before PaTH's path-cumulative
transform, and the raw query slice from standard GPT-2 eager attention.
It then computes per-layer cosine similarity matrices across sequence positions,
summarizes off-diagonal similarity, and produces relative-distance decay curves.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/project/nlp-work5/hongyu-s/transformers/src")
sys.path.insert(0, "/project/nlp-work5/hongyu-s/flash-linear-attention")

import fla.models  # noqa: F401,E402  # registers PaTH attention with HF
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402


WORKDIR = Path("/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling")
OUT_DIR = WORKDIR / "analysis" / "query_similarity"
JSON_OUT = OUT_DIR / "query_sim_path_vs_nope.json"
PNG_OUT = OUT_DIR / "query_sim_path_vs_nope.png"

PATH_CKPT = WORKDIR / "runs/PA_baseline_multi_seeds/token_even_mix_PA_s42/checkpoint-15000"
NOPE_CKPT = WORKDIR / "runs/wikitext_pe_cmp/wavelet/finetune_eager_nope_seed42/checkpoint-15900"

WIKITEXT_CANDIDATES = [
    WORKDIR / "wikitext-103-raw-v1" / "wiki.valid.raw",
    WORKDIR / "wikitext-103-raw-v1" / "valid.txt",
    WORKDIR / "wikitext-103-raw-v1" / "validation.txt",
    WORKDIR / "data" / "wikitext-103-raw-v1" / "wiki.valid.raw",
    WORKDIR / "data" / "wikitext-103-raw-v1" / "valid.txt",
]
HOTPOT_JSONL = WORKDIR / "hotpot_long" / "data" / "hotpot_long_dev_uniform.jsonl"

SEQ_LEN = 512
N_EXAMPLES = 8
N_LAYERS = 12
N_HEADS = 12
HEAD_DIM = 64


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


def _load_eval_batch(tokenizer) -> tuple[torch.Tensor, str]:
    texts, source = _load_texts()
    batch = _chunk_token_ids(tokenizer, texts, SEQ_LEN, N_EXAMPLES)
    return batch, source


def _load_path_model() -> torch.nn.Module:
    config = AutoConfig.from_pretrained(PATH_CKPT)
    config.attn_implementation = "path_attn"
    config.use_cache = False
    return AutoModelForCausalLM.from_pretrained(PATH_CKPT, config=config)


def _load_nope_model() -> torch.nn.Module:
    return AutoModelForCausalLM.from_pretrained(NOPE_CKPT)


def _capture_path_queries(model, batch: torch.Tensor) -> list[torch.Tensor]:
    captured: dict[int, torch.Tensor] = {}
    handles = []

    for layer_idx, block in enumerate(model.transformer.h):
        q_proj = block.attn.core.q_proj

        def make_hook(lid: int):
            def hook(module, inputs, output):
                captured[lid] = output.detach().float().cpu()

            return hook

        handles.append(q_proj.register_forward_hook(make_hook(layer_idx)))

    with torch.no_grad():
        model(input_ids=batch)

    for h in handles:
        h.remove()

    queries = []
    for layer_idx in range(len(model.transformer.h)):
        q = captured[layer_idx].reshape(batch.shape[0], batch.shape[1], N_HEADS, HEAD_DIM)
        queries.append(q)
    return queries


def _capture_nope_queries(model, batch: torch.Tensor) -> list[torch.Tensor]:
    captured: dict[int, torch.Tensor] = {}
    handles = []

    for layer_idx, block in enumerate(model.transformer.h):
        c_attn = block.attn.c_attn

        def make_hook(lid: int):
            def hook(module, inputs, output):
                out = output[0] if isinstance(output, tuple) else output
                captured[lid] = out.detach().float().cpu()

            return hook

        handles.append(c_attn.register_forward_hook(make_hook(layer_idx)))

    with torch.no_grad():
        model(input_ids=batch)

    for h in handles:
        h.remove()

    queries = []
    hidden = N_HEADS * HEAD_DIM
    for layer_idx in range(len(model.transformer.h)):
        qkv = captured[layer_idx]
        q = qkv[..., :hidden]
        q = q.view(batch.shape[0], batch.shape[1], N_HEADS, HEAD_DIM)
        queries.append(q)
    return queries


def _cosine_sim_matrix(q: torch.Tensor) -> torch.Tensor:
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.einsum("bthd,bshd->bhts", q, q)


def _summarize_similarity(sim: torch.Tensor) -> tuple[float, np.ndarray]:
    # sim: [B, H, T, T]
    sim_np = sim.mean(dim=0).numpy()  # [H, T, T]
    offdiag_mask = ~np.eye(sim_np.shape[-1], dtype=bool)
    mean_offdiag = float(sim_np[:, offdiag_mask].mean())

    T = sim_np.shape[-1]
    dist_curve = np.zeros(T, dtype=np.float64)
    counts = np.zeros(T, dtype=np.int64)
    for d in range(T):
        vals = []
        for i in range(T - d):
            j = i + d
            vals.append(sim_np[:, i, j])
            if d != 0:
                vals.append(sim_np[:, j, i])
        vals = np.asarray(vals, dtype=np.float64).reshape(-1)
        dist_curve[d] = float(vals.mean())
        counts[d] = vals.size
    return mean_offdiag, dist_curve


def _analyze_model(name: str, queries: list[torch.Tensor]) -> dict:
    layer_entries = []
    layer_means = []
    layer_curves = []
    for layer_idx, q in enumerate(queries):
        sim = _cosine_sim_matrix(q)
        mean_offdiag, dist_curve = _summarize_similarity(sim)
        layer_entries.append(
            {
                "layer": layer_idx,
                "mean_offdiag_cos_sim": mean_offdiag,
                "dist_curve": dist_curve.tolist(),
            }
        )
        layer_means.append(mean_offdiag)
        layer_curves.append(dist_curve)

    return {
        "layers": layer_entries,
        "overall_mean_offdiag_cos_sim": float(np.mean(layer_means)),
        "overall_dist_curve": np.mean(np.stack(layer_curves, axis=0), axis=0).tolist(),
        "name": name,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    batch, source = _load_eval_batch(tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = batch.to(device)

    path_model = _load_path_model().eval().to(device)
    nope_model = _load_nope_model().eval().to(device)

    path_queries = _capture_path_queries(path_model, batch)
    nope_queries = _capture_nope_queries(nope_model, batch)

    path_res = _analyze_model("path_only", path_queries)
    nope_res = _analyze_model("nope", nope_queries)

    payload = {
        "source": source,
        "seq_len": SEQ_LEN,
        "n_examples": N_EXAMPLES,
        "path_only": path_res,
        "nope": nope_res,
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    x = np.arange(N_LAYERS)
    width = 0.38
    axes[0].bar(x - width / 2, [l["mean_offdiag_cos_sim"] for l in path_res["layers"]], width, label="PaTH-only")
    axes[0].bar(x + width / 2, [l["mean_offdiag_cos_sim"] for l in nope_res["layers"]], width, label="NoPE")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Mean off-diagonal cosine sim")
    axes[0].set_title("Per-layer raw query similarity")
    axes[0].set_xticks(x)
    axes[0].legend()
    axes[0].grid(True, axis="y", alpha=0.25)

    d = np.arange(SEQ_LEN)
    axes[1].plot(d, np.array(path_res["overall_dist_curve"]), label="PaTH-only")
    axes[1].plot(d, np.array(nope_res["overall_dist_curve"]), label="NoPE")
    axes[1].set_xlabel("Relative distance |i-j|")
    axes[1].set_ylabel("Mean cosine similarity")
    axes[1].set_title("Layer-averaged distance decay")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    fig.suptitle("Raw query cosine similarity across positions")
    fig.tight_layout()
    fig.savefig(PNG_OUT, dpi=160)

    path_mean = path_res["overall_mean_offdiag_cos_sim"]
    nope_mean = nope_res["overall_mean_offdiag_cos_sim"]
    diff = path_mean - nope_mean
    winner = "PaTH-only" if diff > 0 else "NoPE"
    print(f"source={source}")
    print(f"PaTH-only overall mean off-diagonal cosine similarity: {path_mean:.6f}")
    print(f"NoPE overall mean off-diagonal cosine similarity:     {nope_mean:.6f}")
    print(f"Higher: {winner} by {abs(diff):.6f}")


if __name__ == "__main__":
    main()

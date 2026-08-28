#!/usr/bin/env python3
"""NMF motif decomposition of PaTH logits, fit independently per layer."""

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from sklearn.decomposition import NMF as SklearnNMF
from transformers import AutoTokenizer

from run_head_ablation_eval import build_capture_input_ids, find_path_attention_modules, load_path_attn_model


def _git_sha(repo_path: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", repo_path, "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


_NMF_MAX_NORMALIZED = 5.0
_NMF_MIN_VALID_KEYS = 4


def robust_relu_normalize(logit_mat: torch.Tensor) -> torch.Tensor:
    T = logit_mat.shape[0]
    out = torch.zeros(T, T, dtype=torch.float32)
    for i in range(T):
        if i + 1 < _NMF_MIN_VALID_KEYS:
            continue
        vals = logit_mat[i, :i + 1].float()
        m_i = float(vals.median().item())
        mad = float((vals - m_i).abs().median().item())
        s_i = 1.4826 * mad + 1e-6
        out[i, :i + 1] = (vals - m_i).div(s_i).clamp_(0.0, _NMF_MAX_NORMALIZED)
    return out


def masked_sum_pool_2d(X_raw: torch.Tensor, M: int = 128) -> torch.Tensor:
    T = X_raw.shape[0]
    row_bin = torch.arange(T) * M // T
    col_bin = torch.arange(T) * M // T
    fi = torch.arange(T).unsqueeze(1).expand(T, T).reshape(-1)
    fj = torch.arange(T).unsqueeze(0).expand(T, T).reshape(-1)
    valid = fj <= fi
    ri = row_bin.unsqueeze(1).expand(T, T).reshape(-1)
    ci = col_bin.unsqueeze(0).expand(T, T).reshape(-1)
    flat_idx = (ri * M + ci)[valid]
    flat_vals = X_raw.reshape(-1)[valid]
    out = torch.zeros(M * M, dtype=torch.float32)
    out.scatter_add_(0, flat_idx, flat_vals)
    return out.reshape(M, M)


def salient_mass_filter(pool_map: torch.Tensor, alpha: float = 0.90) -> torch.Tensor:
    M = pool_map.shape[0]
    out = torch.zeros_like(pool_map)
    for r in range(M):
        row = pool_map[r, :r + 1]
        row_sum = row.sum().item()
        if row_sum < 1e-12:
            continue
        sorted_vals, _ = torch.sort(row, descending=True)
        cumsum = torch.cumsum(sorted_vals, dim=0)
        nz = (cumsum >= alpha * row_sum).nonzero(as_tuple=False)
        if nz.numel() == 0:
            out[r, :r + 1] = row
            continue
        k = int(nz[0].item())
        thresh = float(sorted_vals[k].item()) if k < sorted_vals.shape[0] else 0.0
        out[r, :r + 1] = row * (row >= thresh).float()
    return out


def top_pct_filter(pool_map: torch.Tensor, keep_frac: float) -> torch.Tensor:
    M = pool_map.shape[0]
    out = torch.zeros_like(pool_map)
    for r in range(M):
        row = pool_map[r, :r + 1]
        pos = row[row > 0]
        if pos.numel() == 0:
            continue
        thresh = float(torch.quantile(pos, 1.0 - keep_frac).item())
        out[r, :r + 1] = row * (row >= thresh).float()
    return out


def l1_normalize_map(pool_map: torch.Tensor) -> torch.Tensor:
    s = pool_map.sum().item()
    if s < 1e-12:
        return torch.zeros_like(pool_map)
    return pool_map / s


def preprocess_logit_map(logit_mat: torch.Tensor, M: int = 128, preprocessing: str = "salient") -> Optional[torch.Tensor]:
    X_raw = robust_relu_normalize(logit_mat)
    pool = masked_sum_pool_2d(X_raw, M=M)
    if preprocessing == "salient":
        pool = salient_mass_filter(pool, alpha=0.90)
    elif preprocessing == "top10":
        pool = top_pct_filter(pool, keep_frac=0.10)
    elif preprocessing == "top20":
        pool = top_pct_filter(pool, keep_frac=0.20)
    pool = l1_normalize_map(pool)
    if pool.sum().item() < 1e-12:
        return None
    return pool.reshape(-1).float()


def render_doc(title: str, sentences: list) -> str:
    return f"Title: {title}\n{' '.join(sentences).strip()}\n\n"


def render_prompt(ex: dict) -> str:
    ctx = "".join(render_doc(t, s) for t, s in ex["context"])
    return f"Question: {ex['question']}\n\nContext:\n{ctx}Answer:"


def load_cases(jsonl_path: Path, seq_len: int, n_case: int):
    cases = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["meta"].get("target_total_tokens") == seq_len:
                cases.append(rec)
                if len(cases) >= n_case:
                    break
    return cases


def load_cases_by_length(jsonl_path: Path, seq_len: int) -> list:
    cases = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["meta"].get("target_total_tokens") == seq_len:
                cases.append(rec)
    return cases


def fit_layer_dictionary(rows: list, meta_rows: list, rank: int, out_dir: Path, checkpoint: str, seq_len: int, layer_idx: int, M: int,
                         preprocessing: str, capture_total: bool) -> dict:
    if not rows:
        raise RuntimeError(f"No rows for layer {layer_idx:02d}")
    X = np.stack(rows, axis=0).astype(np.float32)
    nmf = SklearnNMF(n_components=rank, init="nndsvda", max_iter=500, random_state=42, verbose=1)
    U = nmf.fit_transform(X)
    V = nmf.components_
    recon_err = float(nmf.reconstruction_err_)

    layer_dir = out_dir / f"layer_{layer_idx:02d}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    np.save(layer_dir / "U.npy", U.astype(np.float32))
    np.save(layer_dir / "V.npy", V.astype(np.float32))
    meta = {
        "checkpoint": checkpoint,
        "seq_len": seq_len,
        "layer_idx": layer_idx,
        "rank": rank,
        "pool_size": M,
        "preprocessing": preprocessing,
        "capture_total": capture_total,
        "n_maps": len(rows),
        "reconstruction_err": recon_err,
        "git_sha_transformers": _git_sha(str(Path(__file__).parents[3])),
    }
    with open(layer_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        basis_dir = layer_dir / "basis_heatmaps"
        basis_dir.mkdir(exist_ok=True)
        for r in range(rank):
            basis = V[r].reshape(M, M)
            vmax = float(np.quantile(basis, 0.995))
            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(basis, cmap="viridis", vmin=0, vmax=max(vmax, 1e-8))
            ax.set_title(f"Layer {layer_idx:02d} basis {r}", fontsize=9)
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.savefig(basis_dir / f"basis_{r:02d}.png", dpi=80)
            plt.close(fig)
    except Exception as e:
        print(f"Warning: layer {layer_idx:02d} heatmap export failed: {e}")

    usage_rows = []
    meta_arr = np.array([[m["case_id"], m["layer"], m["head"]] for m in meta_rows], dtype=object)
    for head in range(12):
        mask = meta_arr[:, 2] == head
        if mask.sum() == 0:
            continue
        U_sub = U[mask]
        for r in range(rank):
            usage_rows.append({
                "layer": layer_idx,
                "head": head,
                "component": r,
                "mean_usage": float(U_sub[:, r].mean()),
                "std_usage": float(U_sub[:, r].std()),
                "n_maps": int(mask.sum()),
            })
    with open(layer_dir / "usage_by_layer_head.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "head", "component", "mean_usage", "std_usage", "n_maps"])
        writer.writeheader()
        writer.writerows(usage_rows)

    return {"U": U, "V": V, "meta": meta, "layer_dir": layer_dir, "recon_err": recon_err}


def run(args):
    checkpoint = str(args.checkpoint)
    seq_len = int(args.seq_len)
    n_case = int(args.n_case)
    rank = int(args.rank)
    M = int(args.pool_size)
    preprocessing = args.preprocessing
    jsonl_path = Path(args.jsonl)
    out_root = Path(args.out_root)
    capture_total = bool(args.capture_total)
    run_suffix = args.run_suffix or ""
    assert preprocessing in ("salient", "dense", "top10", "top20"), preprocessing

    cases = load_cases(jsonl_path, seq_len, n_case)
    print(f"Loaded {len(cases)} cases for seq_len={seq_len}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {checkpoint} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
    model = load_path_attn_model(checkpoint, device)

    path_attn_layers = find_path_attention_modules(model)
    n_layers = len(path_attn_layers)
    if n_layers == 0:
        raise RuntimeError("No PaTHAttention layers found.")
    head_dim = model.config.n_embd // model.config.n_head
    head_dim_scale = head_dim ** -0.5
    print(f"Found {n_layers} PaTHAttention layers")

    ckpt_tag = Path(checkpoint).name
    cap_tag = "total" if capture_total else "paonly"
    run_tag = f"L{seq_len}_{ckpt_tag}_{preprocessing}_R{rank}_{cap_tag}{run_suffix}_layerwise"
    out_dir = out_root / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    layer_rows = {layer_idx: [] for layer_idx in range(n_layers)}
    row_meta = {layer_idx: [] for layer_idx in range(n_layers)}

    for case_idx, rec in enumerate(cases):
        prompt_str = render_prompt(rec)
        answer_str = f" {rec['answer']}"
        prompt_ids = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer_str, add_special_tokens=False)["input_ids"]
        full_ids = (prompt_ids + answer_ids)[:seq_len]
        actual_len = len(full_ids)
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            _ = model(input_ids)

        for layer_idx, module in path_attn_layers:
            attr_name = "_last_logits_full" if capture_total else "_last_logits_pa_only"
            buf = getattr(module, attr_name, None)
            if buf is None:
                raise RuntimeError(f"Layer {layer_idx:02d} missing {attr_name} after forward pass.")
            attn = (buf[0, :, :actual_len, :actual_len].to(torch.float32) * head_dim_scale).cpu()
            n_heads = attn.shape[0]
            for h in range(n_heads):
                row = preprocess_logit_map(attn[h], M=M, preprocessing=preprocessing)
                if row is None:
                    continue
                layer_rows[layer_idx].append(row.numpy())
                row_meta[layer_idx].append({
                    "case_id": case_idx,
                    "layer": layer_idx,
                    "head": h,
                    "actual_len": actual_len,
                    "seq_len": seq_len,
                })
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"Total maps: {sum(len(v) for v in layer_rows.values())}")
    for layer_idx in range(n_layers):
        print(f"  layer {layer_idx:02d}: {len(layer_rows[layer_idx])} maps")
        fit_layer_dictionary(
            layer_rows[layer_idx], row_meta[layer_idx], rank, out_dir, checkpoint, seq_len, layer_idx, M, preprocessing, capture_total
        )
        with open(out_dir / f"layer_{layer_idx:02d}" / "row_meta.json", "w", encoding="utf-8") as f:
            json.dump(row_meta[layer_idx], f)

    meta = {
        "checkpoint": checkpoint,
        "seq_len": seq_len,
        "n_case": len(cases),
        "rank": rank,
        "pool_size": M,
        "preprocessing": preprocessing,
        "capture_total": capture_total,
        "n_layers": n_layers,
        "git_sha_transformers": _git_sha(str(Path(__file__).parents[3])),
        "run_tag": run_tag,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Done. Output: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Per-layer NMF motif decomposition of PaTH logits")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--n_case", type=int, default=20)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--pool_size", type=int, default=128)
    parser.add_argument("--preprocessing", default="salient", choices=["salient", "dense", "top10", "top20"])
    parser.add_argument("--capture_total", action="store_true")
    parser.add_argument("--run_suffix", default="")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

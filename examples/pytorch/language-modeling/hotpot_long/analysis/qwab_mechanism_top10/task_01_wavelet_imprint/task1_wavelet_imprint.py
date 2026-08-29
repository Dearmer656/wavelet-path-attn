#!/usr/bin/env python3
"""PAT-254 Task 1: wavelet-subspace imprint in the trained PaTH backbone.

Question: does QWAB training cause PaTH's OWN logits (online branch off, i.e.
Q-off) to internalize a wavelet-aligned absolute-key component, versus an
independently-trained PA-only checkpoint?

D = A^{Q-off} - A^{PA}   (mean over heads, causal-masked, per layer)

R_wav = ||D P_Psi||_F^2 / ||D||_F^2, where P_Psi is the projector onto the
real (Preflight-validated) causal wavelet template's column space (key axis).
Controls: equal-rank random orthonormal subspace, DCT low-frequency subspace,
position-shuffled Psi -- same rank, same causal support.

First-pass simplification (r=1): Preflight found the real online field is a
near-rank-1 "shared row pattern" (median row cosine ~0.99, effective rank
~1.3), so a single canonical causally-truncated Ricker template (using the
MEDIAN of the real captured per-query shift beta_i, and the checkpoint's own
scale) is a reasonable first Psi. If R_wav is borderline, widen Psi's rank
using multiple representative per-query-shifted templates.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.fft import dct

import sys

sys.path.insert(0, "/project/nlp-work5/hongyu-s/flash-linear-attention")
sys.path.insert(0, "/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long")

from fla.layers.path_attn import PaTHAttention  # noqa: F401
from dump_router_usage import render_prompt_hotpot, load_cases_hotpot, get_path_layers, load_path_attn_model
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def ricker(u: torch.Tensor) -> torch.Tensor:
    return (1.0 - u.pow(2)) * torch.exp(-0.5 * u.pow(2))


def causal_mask(T: int, device) -> torch.Tensor:
    i_idx = torch.arange(T, device=device).view(T, 1)
    j_idx = torch.arange(T, device=device).view(1, T)
    return (j_idx <= i_idx).to(torch.float32)


def build_psi_real(T: int, beta_median: float, scale: float, device) -> torch.Tensor:
    diff = torch.arange(T, device=device, dtype=torch.float32)
    u = (diff - beta_median) / max(scale, 1e-6)
    psi = ricker(u)  # [T]
    return psi.view(T, 1)  # [T, r=1]


def build_psi_random(T: int, r: int, device, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    mat = torch.randn(T, r, generator=g).to(device)
    q, _ = torch.linalg.qr(mat)
    return q[:, :r]


def build_psi_dct(T: int, r: int, device) -> torch.Tensor:
    basis = dct(np.eye(T), axis=0, norm="ortho")[:, :r]
    return torch.tensor(basis, dtype=torch.float32, device=device)


def build_psi_shuffled(psi_real: torch.Tensor, seed: int) -> torch.Tensor:
    g = np.random.default_rng(seed)
    perm = g.permutation(psi_real.shape[0])
    return psi_real[perm]


def projector(psi: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    r = psi.shape[1]
    gram = psi.T @ psi + eps * torch.eye(r, device=psi.device)
    return psi @ torch.linalg.inv(gram) @ psi.T


def r_wav(D: torch.Tensor, P: torch.Tensor) -> float:
    proj = D @ P
    num = float((proj ** 2).sum().item())
    den = float((D ** 2).sum().item()) + 1e-12
    return num / den


def capture_A(module) -> torch.Tensor:
    pa = getattr(module, "_last_logits_pa_only")
    return pa[0].mean(dim=0)  # [T,T], mean over heads


def enable_do_null(path_layers):
    for _, m in path_layers:
        m.wavelet_intervention_enable = True
        m.wavelet_intervention_mode = "ctxscale_null"
        m.wavelet_intervention_strict = True
        m._wavelet_intervention_targets = {m.layer_idx: "all"}


def disable_do_null(path_layers):
    for _, m in path_layers:
        m.wavelet_intervention_enable = False


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.pa_checkpoint, use_fast=False)

    print(f"Loading PA-only: {args.pa_checkpoint}")
    pa_model = load_path_attn_model(args.pa_checkpoint, device, dtype=torch.float32)
    pa_layers = get_path_layers(pa_model)
    for _, m in pa_layers:
        m._capture_debug_tensors = True

    print(f"Loading QWAB: {args.qwab_checkpoint}")
    qwab_model = load_path_attn_model(args.qwab_checkpoint, device, dtype=torch.float32)
    qwab_layers = get_path_layers(qwab_model)
    for _, m in qwab_layers:
        m._capture_debug_tensors = True

    n_layers = len(pa_layers)
    assert n_layers == len(qwab_layers), (n_layers, len(qwab_layers))

    cases = load_cases_hotpot(args.jsonl, args.seq_len, args.n_case)
    print(f"Loaded {len(cases)} cases @ L={args.seq_len}")

    cmask = causal_mask(args.seq_len, device)
    n_random = args.n_random_controls

    per_case_rows = []
    for ci, rec in enumerate(cases):
        prompt_ids = tokenizer(render_prompt_hotpot(rec), add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(f" {rec['answer']}", add_special_tokens=False)["input_ids"]
        full_ids = (prompt_ids + answer_ids)[: args.seq_len]
        if len(full_ids) < args.seq_len:
            continue
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            _ = pa_model(input_ids)
        A_PA = [capture_A(m) * cmask for _, m in pa_layers]

        enable_do_null(qwab_layers)
        with torch.no_grad():
            _ = qwab_model(input_ids)
        A_Qoff = [capture_A(m) * cmask for _, m in qwab_layers]

        disable_do_null(qwab_layers)
        with torch.no_grad():
            _ = qwab_model(input_ids)
        beta_by_layer = []
        scale_by_layer = []
        for _, m in qwab_layers:
            bmap = getattr(m, "_last_ctxscale_beta_by_scale", None)
            scales = getattr(m, "_last_ctxscale_scales", None)
            if bmap is not None and 0 in bmap and scales is not None:
                beta_by_layer.append(float(bmap[0][0].median().item()))
                scale_by_layer.append(float(scales[0].item()))
            else:
                beta_by_layer.append(0.0)
                scale_by_layer.append(128.0)

        for layer_idx in range(n_layers):
            D_raw = A_Qoff[layer_idx] - A_PA[layer_idx]
            # Mean-center over the causally-valid support only: DCT's rank-1
            # control is exactly the constant/DC basis vector, so an
            # off-center D would let ANY rank-1 control (not just DCT)
            # trivially "explain" a generic overall level-shift between PA
            # and Q-off logits, which is a different (and much weaker) claim
            # than "wavelet-shaped". Centering isolates the shape question.
            valid = cmask.bool()
            d_mean = D_raw[valid].mean()
            D = torch.where(valid, D_raw - d_mean, D_raw)
            psi_real = build_psi_real(args.seq_len, beta_by_layer[layer_idx], scale_by_layer[layer_idx], device)
            P_real = projector(psi_real)
            R_real = r_wav(D, P_real)

            R_rand = [r_wav(D, projector(build_psi_random(args.seq_len, 1, device, seed=1000 + s))) for s in range(n_random)]
            P_dct = projector(build_psi_dct(args.seq_len, 1, device))
            R_dct = r_wav(D, P_dct)
            P_shuf = projector(build_psi_shuffled(psi_real, seed=42))
            R_shuf = r_wav(D, P_shuf)

            per_case_rows.append({
                "case_idx": ci, "layer": layer_idx,
                "R_wav_real": R_real,
                "R_wav_random_mean": float(np.mean(R_rand)),
                "R_wav_random_std": float(np.std(R_rand)),
                "R_wav_dct": R_dct,
                "R_wav_shuffled": R_shuf,
                "D_frob_norm": float(D.norm().item()),
                "beta_median": beta_by_layer[layer_idx],
                "scale": scale_by_layer[layer_idx],
            })
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  case {ci + 1}/{len(cases)} done")

    out_csv = OUT_DIR / f"{args.model_tag}_L{args.seq_len}.csv"
    fieldnames = list(per_case_rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(per_case_rows)
    print(f"wrote {out_csv} ({len(per_case_rows)} rows)")

    by_layer = {}
    for r in per_case_rows:
        by_layer.setdefault(r["layer"], []).append(r)
    summary = {"model_tag": args.model_tag, "seq_len": args.seq_len, "n_case": len(cases), "layers": {}}
    rng = np.random.default_rng(0)
    for layer, rows in by_layer.items():
        real = np.array([r["R_wav_real"] for r in rows])
        rand = np.array([r["R_wav_random_mean"] for r in rows])
        diff = real - rand
        boots = np.array([rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(2000)])
        ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
        summary["layers"][int(layer)] = {
            "R_wav_real_mean": float(real.mean()),
            "R_wav_random_mean": float(rand.mean()),
            "R_wav_dct_mean": float(np.mean([r["R_wav_dct"] for r in rows])),
            "R_wav_shuffled_mean": float(np.mean([r["R_wav_shuffled"] for r in rows])),
            "real_minus_random_mean": float(diff.mean()),
            "real_minus_random_ci95": [float(ci_lo), float(ci_hi)],
            "ci_excludes_zero": bool(ci_lo > 0 or ci_hi < 0),
        }
    out_json = OUT_DIR / f"{args.model_tag}_L{args.seq_len}_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out_json}")
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pa_checkpoint", required=True)
    p.add_argument("--qwab_checkpoint", required=True)
    p.add_argument("--model_tag", required=True)
    p.add_argument("--seq_len", type=int, required=True)
    p.add_argument("--n_case", type=int, default=10)
    p.add_argument("--n_random_controls", type=int, default=8)
    p.add_argument("--jsonl", default="/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/hotpot_long_dev.jsonl")
    run(p.parse_args())


if __name__ == "__main__":
    main()

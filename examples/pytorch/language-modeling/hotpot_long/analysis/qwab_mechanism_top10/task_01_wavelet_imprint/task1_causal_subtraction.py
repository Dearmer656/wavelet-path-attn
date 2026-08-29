#!/usr/bin/env python3
"""PAT-254 Task 1, acceptance criterion 2: causal-subtraction ablation.

Fit phase (disjoint "fit pool", N_fit cases): compute per-layer average
D = A^Q-off - A^PA (mean over heads, causal-masked, mean-centered) and its
projection onto (a) the real rank-1 wavelet Psi and (b) the matched DCT-rank1
control. These give two FIXED [T,T] ablation matrices per layer:
  delta_real[l] = D_avg[l] @ P_real[l]
  delta_dct[l]  = D_avg[l] @ P_dct[l]

Eval phase (disjoint "eval pool", N_eval cases, teacher-forced F1 over the
answer span): for lambda in {0, 0.25, 0.5, 1}, run the QWAB checkpoint with
do(null) (Q-off state) PLUS `_ctxscale_subtract_spec` subtracting
lambda*delta[l] at every layer, and measure Delta F1 vs the lambda=0 (pure
Q-off) baseline. Real-Psi subtraction removing MORE F1 than matched-rank
DCT subtraction, especially at long length, is the acceptance signal per the
issue; a flat/no difference corroborates Task 1's correlational finding that
the wavelet-shaped component is a minor, not the dominant, piece of D.
"""
import argparse
import csv
import json
import re
import string
from collections import Counter
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


def ricker(u):
    return (1.0 - u.pow(2)) * torch.exp(-0.5 * u.pow(2))


def causal_mask(T, device):
    i_idx = torch.arange(T, device=device).view(T, 1)
    j_idx = torch.arange(T, device=device).view(1, T)
    return (j_idx <= i_idx).to(torch.float32)


def projector(psi, eps=1e-6):
    r = psi.shape[1]
    gram = psi.T @ psi + eps * torch.eye(r, device=psi.device)
    return psi @ torch.linalg.inv(gram) @ psi.T


def build_psi_dct(T, r, device):
    basis = dct(np.eye(T), axis=0, norm="ortho")[:, :r]
    return torch.tensor(basis, dtype=torch.float32, device=device)


def _normalize(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def _f1(pred, ref):
    pred_toks = _normalize(pred).split()
    ref_toks = _normalize(ref).split()
    if len(pred_toks) == 0 and len(ref_toks) == 0:
        return 1.0
    if len(pred_toks) == 0 or len(ref_toks) == 0:
        return 0.0
    common = Counter(pred_toks) & Counter(ref_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    p = num_same / len(pred_toks)
    r = num_same / len(ref_toks)
    return 2 * p * r / (p + r)


def capture_A(module):
    return module._last_logits_pa_only[0].mean(dim=0)


def set_do_null(path_layers, enabled):
    for _, m in path_layers:
        m.wavelet_intervention_enable = enabled
        m.wavelet_intervention_mode = "ctxscale_null"
        m.wavelet_intervention_strict = True
        m._wavelet_intervention_targets = {m.layer_idx: "all"} if enabled else None


def set_subtract(path_layers, deltas, lam):
    for li, (_, m) in enumerate(path_layers):
        if deltas is None or lam == 0.0:
            m._ctxscale_subtract_spec = {"enabled": False}
        else:
            m._ctxscale_subtract_spec = {"enabled": True, "delta": deltas[li], "lambda": lam}


def fit_deltas(pa_model, pa_layers, qwab_model, qwab_layers, tokenizer, fit_cases, seq_len, device):
    T = seq_len
    cmask = causal_mask(T, device)
    n_layers = len(pa_layers)
    D_sum = [torch.zeros(T, T, device=device) for _ in range(n_layers)]
    beta_sum = [0.0] * n_layers
    scale_val = [128.0] * n_layers
    n_used = 0
    for rec in fit_cases:
        prompt_ids = tokenizer(render_prompt_hotpot(rec), add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(f" {rec['answer']}", add_special_tokens=False)["input_ids"]
        full_ids = (prompt_ids + answer_ids)[:T]
        if len(full_ids) < T:
            continue
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            _ = pa_model(input_ids)
        A_PA = [capture_A(m) * cmask for _, m in pa_layers]
        set_do_null(qwab_layers, True)
        set_subtract(qwab_layers, None, 0.0)
        with torch.no_grad():
            _ = qwab_model(input_ids)
        A_Qoff = [capture_A(m) * cmask for _, m in qwab_layers]
        set_do_null(qwab_layers, False)
        with torch.no_grad():
            _ = qwab_model(input_ids)
        for li, (_, m) in enumerate(qwab_layers):
            bmap = getattr(m, "_last_ctxscale_beta_by_scale", None)
            scales = getattr(m, "_last_ctxscale_scales", None)
            if bmap is not None and 0 in bmap and scales is not None:
                beta_sum[li] += float(bmap[0][0].median().item())
                scale_val[li] = float(scales[0].item())
            D_sum[li] += A_Qoff[li] - A_PA[li]
        n_used += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    assert n_used > 0
    D_avg = [d / n_used for d in D_sum]
    beta_avg = [b / n_used for b in beta_sum]

    deltas_real, deltas_dct = [], []
    diff = torch.arange(T, device=device, dtype=torch.float32)
    for li in range(n_layers):
        D = D_avg[li]
        valid = cmask.bool()
        D = torch.where(valid, D - D[valid].mean(), D)
        u = (diff - beta_avg[li]) / max(scale_val[li], 1e-6)
        psi_real = ricker(u).view(T, 1)
        P_real = projector(psi_real)
        deltas_real.append((D @ P_real).detach())
        P_dct = projector(build_psi_dct(T, 1, device))
        deltas_dct.append((D @ P_dct).detach())
    return deltas_real, deltas_dct, n_used


def eval_f1(qwab_model, qwab_layers, tokenizer, eval_cases, seq_len, device, deltas, lam):
    set_do_null(qwab_layers, True)
    set_subtract(qwab_layers, deltas, lam)
    f1s = []
    for rec in eval_cases:
        prompt_ids = tokenizer(render_prompt_hotpot(rec), add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(f" {rec['answer']}", add_special_tokens=False)["input_ids"]
        # PAT-253 lesson (feedback_hotpot_eval_no_truncate): never truncate
        # prompt+answer to seq_len -- target_total_tokens is the CONTEXT
        # budget, not the full rendered prompt, so the wrapped prompt
        # ("Question:...Context:...Answer:") routinely exceeds seq_len by a
        # few-to-dozens of tokens; slicing to seq_len cuts off "Answer:" and
        # the answer itself. Use the natural (untruncated) length instead --
        # the fixed [seq_len,seq_len] ablation delta gets auto-padded to
        # match by the _ctxscale_subtract_spec hook in path_attn.py.
        full_ids = prompt_ids + answer_ids
        if len(answer_ids) == 0:
            continue
        ans_len = len(answer_ids)
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = qwab_model(input_ids)
        logits = out.logits[0]  # [T, V]
        ans_start = len(full_ids) - ans_len
        pred_ids = logits[ans_start - 1: ans_start - 1 + ans_len].argmax(dim=-1).tolist()
        pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
        gold_text = rec["answer"]
        f1s.append(_f1(pred_text, gold_text))
    set_subtract(qwab_layers, None, 0.0)
    set_do_null(qwab_layers, False)
    return float(np.mean(f1s)) if f1s else float("nan"), len(f1s)


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

    all_cases = load_cases_hotpot(args.jsonl, args.seq_len, args.n_fit + args.n_eval)
    fit_cases = all_cases[: args.n_fit]
    eval_cases = all_cases[args.n_fit: args.n_fit + args.n_eval]
    print(f"fit={len(fit_cases)} eval={len(eval_cases)} @ L={args.seq_len}")

    deltas_real, deltas_dct, n_used = fit_deltas(pa_model, pa_layers, qwab_model, qwab_layers, tokenizer, fit_cases, args.seq_len, device)
    print(f"fit done on {n_used} cases")

    lambdas = [0.0, 0.25, 0.5, 1.0]
    rows = []
    for lam in lambdas:
        f1_real, n1 = eval_f1(qwab_model, qwab_layers, tokenizer, eval_cases, args.seq_len, device, deltas_real, lam)
        print(f"  lambda={lam} real: F1={f1_real:.4f} (n={n1})")
        rows.append({"seq_len": args.seq_len, "control": "real_psi", "lambda": lam, "f1": f1_real, "n_eval": n1})
        if lam > 0:
            f1_dct, n2 = eval_f1(qwab_model, qwab_layers, tokenizer, eval_cases, args.seq_len, device, deltas_dct, lam)
            print(f"  lambda={lam} dct:  F1={f1_dct:.4f} (n={n2})")
            rows.append({"seq_len": args.seq_len, "control": "dct", "lambda": lam, "f1": f1_dct, "n_eval": n2})

    out_csv = OUT_DIR / f"{args.model_tag}_L{args.seq_len}_causal_subtraction.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv}")
    baseline_f1 = rows[0]["f1"]
    summary = {"model_tag": args.model_tag, "seq_len": args.seq_len, "n_fit": n_used, "n_eval": len(eval_cases),
               "baseline_qoff_f1": baseline_f1,
               "rows": rows}
    out_json = OUT_DIR / f"{args.model_tag}_L{args.seq_len}_causal_subtraction_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pa_checkpoint", required=True)
    p.add_argument("--qwab_checkpoint", required=True)
    p.add_argument("--model_tag", required=True)
    p.add_argument("--seq_len", type=int, required=True)
    p.add_argument("--n_fit", type=int, default=10)
    p.add_argument("--n_eval", type=int, default=30)
    p.add_argument("--jsonl", default="/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/hotpot_long_dev.jsonl")
    run(p.parse_args())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""PAT-254 Task 1, acceptance criterion 2: causal-subtraction ablation.

Per-example design (fixes a real cross-example length-alignment bug found in
an earlier fit-pool/eval-pool version: prompt+answer routinely runs a few to
a few dozen tokens LONGER than target_total_tokens once "Question:...
Context:...Answer:" wrapping is added -- target_total_tokens is the CONTEXT
budget, not the full rendered length. A fixed [seq_len,seq_len] delta fit at
target_total_tokens and zero-padded onto a longer eval sequence leaves the
actual answer-token query rows in the zero-padded region, so the ablation
silently never touches the positions that matter. Fix: for EACH example,
build D/Psi/delta at THAT example's own actual (untruncated) length, and
ablate+eval the SAME example -- a per-example causal test, not a
cross-example transplant. Per PAT-253's own eval convention, prompt+answer is
never truncated to seq_len (cuts off "Answer:").

For each example: D = A^{Q-off} - A^{PA} (mean over heads, causal-masked,
mean-centered), Psi_real from the real captured per-query shift/scale
(median beta), Psi_dct = the rank-1 (constant) DCT control, both at this
example's actual T. delta = D @ P_Psi. Then for lambda in {0,0.25,0.5,1},
subtract lambda*delta from Q-off's own logits at every layer and measure
teacher-forced F1 on the answer span, vs the lambda=0 (pure Q-off) baseline.
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


def predict_f1(qwab_model, full_ids, ans_len, tokenizer, gold_text, device):
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = qwab_model(input_ids)
    logits = out.logits[0]
    ans_start = len(full_ids) - ans_len
    pred_ids = logits[ans_start - 1: ans_start - 1 + ans_len].argmax(dim=-1).tolist()
    pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
    return _f1(pred_text, gold_text)


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
    n_layers = len(qwab_layers)

    cases = load_cases_hotpot(args.jsonl, args.seq_len, args.n_case)
    print(f"Loaded {len(cases)} cases (target_total_tokens={args.seq_len})")

    lambdas = [0.0, 0.25, 0.5, 1.0]
    per_case_rows = []
    for ci, rec in enumerate(cases):
        prompt_ids = tokenizer(render_prompt_hotpot(rec), add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(f" {rec['answer']}", add_special_tokens=False)["input_ids"]
        if len(answer_ids) == 0:
            continue
        full_ids = prompt_ids + answer_ids
        ans_len = len(answer_ids)
        T = len(full_ids)
        gold_text = rec["answer"]

        # --- A^PA ---
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            _ = pa_model(input_ids)
        cmask = causal_mask(T, device)
        A_PA = [capture_A(m) * cmask for _, m in pa_layers]

        # --- A^Qoff (do-null) + capture real beta/scale ---
        set_do_null(qwab_layers, True)
        set_subtract(qwab_layers, None, 0.0)
        with torch.no_grad():
            _ = qwab_model(input_ids)
        A_Qoff = [capture_A(m) * cmask for _, m in qwab_layers]

        set_do_null(qwab_layers, False)
        with torch.no_grad():
            _ = qwab_model(input_ids)
        beta_med = []
        scale_val = []
        for _, m in qwab_layers:
            bmap = getattr(m, "_last_ctxscale_beta_by_scale", None)
            scales = getattr(m, "_last_ctxscale_scales", None)
            if bmap is not None and 0 in bmap and scales is not None:
                beta_med.append(float(bmap[0][0].median().item()))
                scale_val.append(float(scales[0].item()))
            else:
                beta_med.append(0.0)
                scale_val.append(128.0)

        diff = torch.arange(T, device=device, dtype=torch.float32)
        deltas_real, deltas_dct = [], []
        valid = cmask.bool()
        for li in range(n_layers):
            D = A_Qoff[li] - A_PA[li]
            D = torch.where(valid, D - D[valid].mean(), D)
            u = (diff - beta_med[li]) / max(scale_val[li], 1e-6)
            psi_real = ricker(u).view(T, 1)
            deltas_real.append((D @ projector(psi_real)).detach())
            psi_dct = torch.full((T, 1), 1.0 / (T ** 0.5), device=device)  # rank-1 DCT = constant basis fn
            deltas_dct.append((D @ projector(psi_dct)).detach())

        # --- ablate + eval, this same example ---
        set_do_null(qwab_layers, True)
        row = {"case_idx": ci, "T": T}
        for lam in lambdas:
            set_subtract(qwab_layers, deltas_real if lam > 0 else None, lam)
            row[f"f1_real_lam{lam}"] = predict_f1(qwab_model, full_ids, ans_len, tokenizer, gold_text, device)
            if lam > 0:
                set_subtract(qwab_layers, deltas_dct, lam)
                row[f"f1_dct_lam{lam}"] = predict_f1(qwab_model, full_ids, ans_len, tokenizer, gold_text, device)
        set_subtract(qwab_layers, None, 0.0)
        set_do_null(qwab_layers, False)

        per_case_rows.append(row)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  case {ci + 1}/{len(cases)}: {row}")

    out_csv = OUT_DIR / f"{args.model_tag}_L{args.seq_len}_causal_subtraction.csv"
    fieldnames = list(per_case_rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(per_case_rows)
    print(f"wrote {out_csv} ({len(per_case_rows)} rows)")

    summary = {"model_tag": args.model_tag, "seq_len": args.seq_len, "n_case": len(per_case_rows)}
    for lam in [0.25, 0.5, 1.0]:
        base = np.array([r["f1_real_lam0.0"] for r in per_case_rows])
        real = np.array([r[f"f1_real_lam{lam}"] for r in per_case_rows])
        dct = np.array([r[f"f1_dct_lam{lam}"] for r in per_case_rows])
        summary[f"lam{lam}"] = {
            "baseline_f1_mean": float(base.mean()),
            "real_f1_mean": float(real.mean()),
            "dct_f1_mean": float(dct.mean()),
            "delta_real": float((real - base).mean()),
            "delta_dct": float((dct - base).mean()),
        }
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
    p.add_argument("--n_case", type=int, default=20)
    p.add_argument("--jsonl", default="/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/hotpot_long_dev.jsonl")
    run(p.parse_args())


if __name__ == "__main__":
    main()

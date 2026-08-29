#!/usr/bin/env python3
"""PAT-254 section 3: re-establish the PA/Q-off/Q-on downstream F1
decomposition on canonical checkpoints, to decide whether to prioritize
training-induced explanations (Tasks 1-3/7/10, if Q-off~=Q-on both beat PA)
or online explanations (Tasks 4/5/8/9, if Delta_online=F1(Qon)-F1(Qoff)>=0.005).

Teacher-forced F1 (same convention as run_clm.py's compute_metrics and
task1_causal_subtraction.py): per example, argmax over the answer-token
logits, decode, compare to gold text with the standard SQuAD/HotpotQA
token-F1. Never truncates prompt+answer to seq_len (cuts off "Answer:").
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


def set_do_null(path_layers, enabled):
    for _, m in path_layers:
        m.wavelet_intervention_enable = enabled
        m.wavelet_intervention_mode = "ctxscale_null"
        m.wavelet_intervention_strict = True
        m._wavelet_intervention_targets = {m.layer_idx: "all"} if enabled else None


def predict_f1(model, full_ids, ans_len, tokenizer, gold_text, device):
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(input_ids)
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
    for _, m in get_path_layers(pa_model):
        m._capture_debug_tensors = False

    print(f"Loading QWAB: {args.qwab_checkpoint}")
    qwab_model = load_path_attn_model(args.qwab_checkpoint, device, dtype=torch.float32)
    qwab_layers = get_path_layers(qwab_model)
    for _, m in qwab_layers:
        m._capture_debug_tensors = False

    cases = load_cases_hotpot(args.jsonl, args.seq_len, args.n_case)
    print(f"Loaded {len(cases)} cases (target_total_tokens={args.seq_len})")

    rows = []
    for ci, rec in enumerate(cases):
        prompt_ids = tokenizer(render_prompt_hotpot(rec), add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(f" {rec['answer']}", add_special_tokens=False)["input_ids"]
        if len(answer_ids) == 0:
            continue
        full_ids = prompt_ids + answer_ids
        ans_len = len(answer_ids)
        gold_text = rec["answer"]

        f1_pa = predict_f1(pa_model, full_ids, ans_len, tokenizer, gold_text, device)
        set_do_null(qwab_layers, True)
        f1_qoff = predict_f1(qwab_model, full_ids, ans_len, tokenizer, gold_text, device)
        set_do_null(qwab_layers, False)
        f1_qon = predict_f1(qwab_model, full_ids, ans_len, tokenizer, gold_text, device)

        rows.append({"case_idx": ci, "seq_len": args.seq_len, "f1_pa": f1_pa, "f1_qoff": f1_qoff, "f1_qon": f1_qon})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  case {ci + 1}/{len(cases)}: PA={f1_pa:.3f} Qoff={f1_qoff:.3f} Qon={f1_qon:.3f}")

    out_csv = OUT_DIR / f"{args.model_tag}_L{args.seq_len}_pa_qoff_qon.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv} ({len(rows)} rows)")

    pa = np.array([r["f1_pa"] for r in rows])
    qoff = np.array([r["f1_qoff"] for r in rows])
    qon = np.array([r["f1_qon"] for r in rows])
    delta_train = qoff - pa
    delta_online = qon - qoff
    rng = np.random.default_rng(0)

    def boot_ci(diff):
        boots = np.array([rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(2000)])
        return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    ci_train = boot_ci(delta_train)
    ci_online = boot_ci(delta_online)
    summary = {
        "model_tag": args.model_tag, "seq_len": args.seq_len, "n_case": len(rows),
        "f1_pa_mean": float(pa.mean()), "f1_qoff_mean": float(qoff.mean()), "f1_qon_mean": float(qon.mean()),
        "delta_train_mean": float(delta_train.mean()), "delta_train_ci95": list(ci_train),
        "delta_online_mean": float(delta_online.mean()), "delta_online_ci95": list(ci_online),
        "prioritize": (
            "training_induced (Tasks 1-3/7/10)" if abs(delta_online.mean()) < 0.005 else
            "online (Tasks 4/5/8/9)" if delta_online.mean() >= 0.005 else
            "both_nontrivial_complete_both_branches"
        ),
    }
    out_json = OUT_DIR / f"{args.model_tag}_L{args.seq_len}_pa_qoff_qon_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pa_checkpoint", required=True)
    p.add_argument("--qwab_checkpoint", required=True)
    p.add_argument("--model_tag", required=True)
    p.add_argument("--seq_len", type=int, required=True)
    p.add_argument("--n_case", type=int, default=30)
    p.add_argument("--jsonl", default="/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/hotpot_long_dev.jsonl")
    run(p.parse_args())


if __name__ == "__main__":
    main()

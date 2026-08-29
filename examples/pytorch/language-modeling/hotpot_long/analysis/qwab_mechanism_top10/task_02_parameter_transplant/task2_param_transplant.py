#!/usr/bin/env python3
"""PAT-254 Task 2: shared-parameter transplantation to localize which PaTH
subsystem carries QWAB's training-induced gain.

Verified (not guessed) state_dict key groups for the small K1 rho128/256
checkpoint family (both PA-only and QWAB-trained have IDENTICAL shapes and
key names for all 5 groups; the only key-set mismatch is `wavelet_k1_gain`,
a vestigial PA-only param unrelated to these groups):
  Theta_H     : attn.core.w_proj.{0,1}.weight, attn.core.bt_proj.{weight,bias}
  Theta_QKVO  : attn.core.{q,k,v,o}_proj.weight
  Theta_MLP   : mlp.c_fc.{weight,bias}, mlp.c_proj.{weight,bias}
  Theta_LN    : ln_1.{weight,bias}, ln_2.{weight,bias} (per-layer) + ln_f (model-level)
  Theta_embed : transformer.wte.weight

Hybrid models: take a BASE checkpoint's full state_dict and overwrite one
group's keys (optionally restricted to a layer-index prefix) with the OTHER
checkpoint's tensors for those exact keys. do(null) is forced ON always
(online QWAB branch disabled per the issue's spec), so any F1 change is
attributable to the transplanted PARAMETERS, not the online wavelet bias.
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
from safetensors.torch import load_file

import sys

sys.path.insert(0, "/project/nlp-work5/hongyu-s/flash-linear-attention")
sys.path.insert(0, "/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long")

from fla.layers.path_attn import PaTHAttention  # noqa: F401
from dump_router_usage import render_prompt_hotpot, load_cases_hotpot, get_path_layers, load_path_attn_model
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GROUP_PATTERNS = {
    "H": [r"attn\.core\.w_proj\.\d+\.weight", r"attn\.core\.bt_proj\.(weight|bias)"],
    "QKVO": [r"attn\.core\.(q|k|v|o)_proj\.weight"],
    "MLP": [r"mlp\.c_(fc|proj)\.(weight|bias)"],
    "LN": [r"ln_1\.(weight|bias)", r"ln_2\.(weight|bias)", r"^transformer\.ln_f\.(weight|bias)$"],
    "embed": [r"^transformer\.wte\.weight$"],
}


def layer_idx_of_key(key: str):
    m = re.search(r"transformer\.h\.(\d+)\.", key)
    return int(m.group(1)) if m else None


def matches_group(key: str, group: str) -> bool:
    return any(re.search(pat, key) for pat in GROUP_PATTERNS[group])


def build_hybrid_state_dict(base_sd, donor_sd, group: str, layer_max=None):
    hybrid = dict(base_sd)
    replaced = []
    for k in donor_sd:
        if not matches_group(k, group):
            continue
        li = layer_idx_of_key(k)
        if layer_max is not None and li is not None and li >= layer_max:
            continue  # layer-prefix swap: only replace layers < layer_max
        if k not in base_sd:
            continue
        assert base_sd[k].shape == donor_sd[k].shape, (k, base_sd[k].shape, donor_sd[k].shape)
        hybrid[k] = donor_sd[k].clone()
        replaced.append(k)
    return hybrid, replaced


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


def set_do_null(path_layers, enabled=True):
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


def eval_model_f1(model, path_layers, tokenizer, cases, device):
    set_do_null(path_layers, True)
    f1s = []
    for rec in cases:
        prompt_ids = tokenizer(render_prompt_hotpot(rec), add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(f" {rec['answer']}", add_special_tokens=False)["input_ids"]
        if len(answer_ids) == 0:
            continue
        full_ids = prompt_ids + answer_ids
        f1s.append(predict_f1(model, full_ids, len(answer_ids), tokenizer, rec["answer"], device))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return float(np.mean(f1s)) if f1s else float("nan"), len(f1s)


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.pa_checkpoint, use_fast=False)

    pa_sd = load_file(f"{args.pa_checkpoint}/model.safetensors")
    qwab_sd = load_file(f"{args.qwab_checkpoint}/model.safetensors")

    cases = load_cases_hotpot(args.jsonl, args.seq_len, args.n_case)
    print(f"Loaded {len(cases)} cases @ L={args.seq_len}")

    # Base model instances -- we reuse ONE loaded model object per base type and
    # just load_state_dict() a fresh hybrid dict into it before each eval, to
    # avoid reconstructing from_pretrained repeatedly.
    print("Loading PA-host model instance...")
    pa_host = load_path_attn_model(args.pa_checkpoint, device, dtype=torch.float32)
    pa_host_layers = get_path_layers(pa_host)
    for _, m in pa_host_layers:
        m._capture_debug_tensors = False

    print("Loading QWAB-host model instance...")
    qwab_host = load_path_attn_model(args.qwab_checkpoint, device, dtype=torch.float32)
    qwab_host_layers = get_path_layers(qwab_host)
    for _, m in qwab_host_layers:
        m._capture_debug_tensors = False

    rows = []

    def eval_hybrid(host_model, host_layers, base_sd, donor_sd, group, layer_max, tag):
        hybrid_sd, replaced = build_hybrid_state_dict(base_sd, donor_sd, group, layer_max)
        missing, unexpected = host_model.load_state_dict(hybrid_sd, strict=False)
        real_missing = [k for k in missing if "wavelet_k1_gain" not in k and k != "lm_head.weight"]
        assert not real_missing, real_missing
        f1, n = eval_model_f1(host_model, host_layers, tokenizer, cases, device)
        print(f"  {tag}: group={group} layer_max={layer_max} n_replaced={len(replaced)} F1={f1:.4f} (n={n})")
        rows.append({"tag": tag, "group": group, "layer_max": layer_max, "n_replaced": len(replaced), "f1": f1, "n_eval": n})
        return f1

    # Pure baselines (no transplant) for reference.
    print("Baselines...")
    pa_host.load_state_dict(pa_sd, strict=False)
    f1_pa, n_pa = eval_model_f1(pa_host, pa_host_layers, tokenizer, cases, device)
    print(f"  PA baseline (no transplant): F1={f1_pa:.4f} (n={n_pa})")
    rows.append({"tag": "PA_baseline", "group": "none", "layer_max": None, "n_replaced": 0, "f1": f1_pa, "n_eval": n_pa})

    qwab_host.load_state_dict(qwab_sd, strict=False)
    f1_qoff, n_qoff = eval_model_f1(qwab_host, qwab_host_layers, tokenizer, cases, device)
    print(f"  QWAB(do-null) baseline (no transplant): F1={f1_qoff:.4f} (n={n_qoff})")
    rows.append({"tag": "Qoff_baseline", "group": "none", "layer_max": None, "n_replaced": 0, "f1": f1_qoff, "n_eval": n_qoff})

    layer_maxes = [None] if not args.layer_maxes else [None] + [int(x) for x in args.layer_maxes.split(",")]
    for group in args.groups.split(","):
        for lm in layer_maxes:
            # direction 1: PA host + QWAB-trained group
            eval_hybrid(pa_host, pa_host_layers, pa_sd, qwab_sd, group, lm, "PAhost_plus_QWABgroup")
            pa_host.load_state_dict(pa_sd, strict=False)  # reset
            # direction 2: QWAB host + PA group
            eval_hybrid(qwab_host, qwab_host_layers, qwab_sd, pa_sd, group, lm, "QWABhost_plus_PAgroup")
            qwab_host.load_state_dict(qwab_sd, strict=False)  # reset

    out_csv = OUT_DIR / f"{args.model_tag}_L{args.seq_len}_transplant.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv}")
    print(json.dumps(rows, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pa_checkpoint", required=True)
    p.add_argument("--qwab_checkpoint", required=True)
    p.add_argument("--model_tag", required=True)
    p.add_argument("--seq_len", type=int, required=True)
    p.add_argument("--n_case", type=int, default=20)
    p.add_argument("--groups", default="H,QKVO,MLP,LN,embed")
    p.add_argument("--layer_maxes", default="", help="comma-separated layer-prefix cutoffs, e.g. 3,6,9 (full-model swap always included)")
    p.add_argument("--jsonl", default="/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/hotpot_long_dev.jsonl")
    run(p.parse_args())


if __name__ == "__main__":
    main()

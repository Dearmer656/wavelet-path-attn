#!/usr/bin/env python3
"""eval_nope_vertical.py — PAT-203.

Test the hypothesis that the RoPE GPT-2's extrapolation drop is caused by its
vertical attention patterns collapsing beyond training length, by switching OFF
rotary PE (NoPE) on a chosen head set at INFERENCE TIME (no retraining) and
measuring extrapolation performance + the vertical heads' Vertical Score.

The intervention is a runtime monkey-patch of each layer's
`rotary_emb.rotate_queries_or_keys`: rotate all heads, then restore the
un-rotated q/k for the masked heads (q/k are [B,H,T,D] there). No edit to
modeling_gpt2.py.

Per (length L, condition) and per case we record:
  ppl_full     mean token NLL over the whole sequence  (-> perplexity)
  ppl_answer   mean token NLL over the gold answer span (task-relevant LM)
  vs_vertical  mean VS_exc of the 8 PAT-202 vertical heads (pattern strength)
  f1, em       HotpotQA answer via greedy generate vs gold aliases

Conditions (RoPE switched OFF on...):
  none      baseline (RoPE on all heads)
  vertical  the 8 PAT-202 stable vertical heads
  random    8 matched non-vertical heads (specificity control)
  all       full NoPE (reference)

Output: nope_eval_rows_{preset}.parquet  (per-case rows; analyzer concatenates).
"""

import argparse
import collections
import json
import re
import string
import unicodedata
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# 8 stable vertical heads from PAT-202 (layer, head)
VERTICAL_HEADS = [(10, 9), (10, 4), (10, 7), (8, 8), (11, 1), (9, 8), (11, 4), (9, 5)]


# ── prompt rendering (matches eval_hotpot_long.py render_input) ───────────────

def render_doc(title: str, sentences: list) -> str:
    return f"Title: {title}\n{' '.join(sentences).strip()}\n\n"


def render_prompt(ex: dict) -> str:
    ctx = "".join(render_doc(t, s) for t, s in ex["context"])
    return f"Question: {ex['question']}\n\nContext:\n{ctx}Answer:"


# ── HotpotQA F1/EM (standard normalisation) ──────────────────────────────────

def normalize_answer(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.lower()
    s = "".join(ch if ch not in set(string.punctuation) else " " for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _f1(pred: str, gt: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gt).split()
    if not p or not g:
        return float(p == g)
    common = collections.Counter(p) & collections.Counter(g)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    prec, rec = ns / len(p), ns / len(g)
    return 2 * prec * rec / (prec + rec)


def best_f1_em(pred: str, answers: list) -> Tuple[float, float]:
    if not answers:
        return 0.0, 0.0
    f1 = max(_f1(pred, a) for a in answers)
    em = max(float(normalize_answer(pred) == normalize_answer(a)) for a in answers)
    return f1, em


# ── NoPE monkey-patch ────────────────────────────────────────────────────────

def make_masked_rotate(orig_rotate, nope_mask: torch.Tensor):
    """Wrap rotate_queries_or_keys so masked heads keep their UN-rotated q/k.

    nope_mask: [H] float, 1.0 = NoPE (skip rotation) for that head.
    Applies when the input is [B,H,T,D] (head axis == len(mask)); pass-through
    otherwise (defensive).
    """
    H = int(nope_mask.numel())

    def wrapped(t, *a, **kw):
        rot = orig_rotate(t, *a, **kw)
        if t.dim() == 4 and t.shape[1] == H:
            m = nope_mask.to(device=rot.device, dtype=rot.dtype).view(1, H, 1, 1)
            return rot * (1.0 - m) + t * m
        return rot
    return wrapped


def apply_nope(model, head_set, n_layer, n_head):
    """Install the NoPE patch; head_set = set of (layer, head) to switch RoPE off."""
    per_layer = {l: torch.zeros(n_head) for l in range(n_layer)}
    for (l, h) in head_set:
        per_layer[l][h] = 1.0
    n_patched = 0
    for name, module in model.named_modules():
        re_m = re.search(r"\.h\.(\d+)\.attn$", name)
        if re_m is None or not hasattr(module, "rotary_emb"):
            continue
        layer = int(re_m.group(1))
        if per_layer[layer].sum() == 0:
            continue
        re_emb = module.rotary_emb
        if getattr(re_emb, "_pat203_patched", False):
            continue
        re_emb.rotate_queries_or_keys = make_masked_rotate(
            re_emb.rotate_queries_or_keys, per_layer[layer])
        re_emb._pat203_patched = True
        n_patched += 1
    return n_patched


# ── frequency-subset NoPE: switch RoPE off only on LOW-FREQ (long-period) dims ──

def make_dim_masked_rotate(orig_rotate, dim_mask: torch.Tensor):
    """Wrap rotate_queries_or_keys so masked DIMS keep their UN-rotated q/k.

    dim_mask: [D] float, 1.0 = NoPE (skip rotation) for that feature dim. Applied
    on the last axis, so it affects ALL heads of the layer identically. This makes
    the selected 2D RoPE pairs behave as NoPE (identity rotation) while the rest
    keep standard RoPE — i.e. frequency-truncated RoPE.
    """
    D = int(dim_mask.numel())

    def wrapped(t, *a, **kw):
        rot = orig_rotate(t, *a, **kw)
        if t.shape[-1] == D:
            m = dim_mask.to(device=rot.device, dtype=rot.dtype)
            return rot * (1.0 - m) + t * m
        return rot
    return wrapped


def lowfreq_dim_mask(freqs: torch.Tensor, head_dim: int, period_cutoff: float):
    """Build a [head_dim] NoPE mask = 1.0 on dims of 2D pairs whose RoPE period
    P_m = 2*pi/theta_m exceeds period_cutoff (the period-underexposed long-period
    dims, per PAT-209 step-1). Interleaved convention: pair m -> dims (2m, 2m+1).
    """
    inv = freqs.detach().float().cpu().numpy()        # theta_m, length head_dim//2
    import numpy as _np
    periods = 2.0 * _np.pi / _np.clip(inv, 1e-12, None)
    mask = torch.zeros(head_dim)
    n_pairs = head_dim // 2
    n_off = 0
    for mi in range(min(n_pairs, len(periods))):
        if periods[mi] > period_cutoff:
            mask[2 * mi] = 1.0
            mask[2 * mi + 1] = 1.0
            n_off += 1
    return mask, n_off, periods


def apply_lowfreq_nope(model, layers, period_cutoff, n_layer, n_head):
    """Switch RoPE off ONLY on the low-freq (P>period_cutoff) dims, on the given
    layers. Returns (n_layers_patched, n_pairs_off, periods)."""
    layers = set(layers)
    n_patched, n_off, periods = 0, 0, None
    for name, module in model.named_modules():
        re_m = re.search(r"\.h\.(\d+)\.attn$", name)
        if re_m is None or not hasattr(module, "rotary_emb"):
            continue
        layer = int(re_m.group(1))
        if layer not in layers:
            continue
        re_emb = module.rotary_emb
        if getattr(re_emb, "_pat203_patched", False):
            continue
        head_dim = int(model.config.n_embd // n_head)
        mask, n_off, periods = lowfreq_dim_mask(re_emb.freqs, head_dim, period_cutoff)
        re_emb.rotate_queries_or_keys = make_dim_masked_rotate(
            re_emb.rotate_queries_or_keys, mask)
        re_emb._pat203_patched = True
        n_patched += 1
    return n_patched, n_off, periods


# ── vertical score (PAT-202 metric, vertical heads only) ─────────────────────

def vs_exc_for_heads(attn_layer: torch.Tensor, heads: List[int], topk=4,
                     gap=16, min_queries=64) -> List[float]:
    """VS_exc (excl. j=0 BOS sink) for the given heads of one layer's attn [H,T,T]."""
    A = attn_layer.float()
    H, T, _ = A.shape
    j_max = max(1, T - gap - min_queries)
    jr = torch.arange(j_max + 1, device=A.device)
    rows = jr + gap
    revcum = A.flip(1).cumsum(1).flip(1)
    far_sum = revcum[:, rows, jr]
    counts = (T - rows).clamp(min=1).float()
    c = far_sum / counts.unsqueeze(0)
    c_ns = c[:, 1:]
    total_ns = c_ns.sum(dim=-1).clamp(min=1e-12)
    k_ns = min(topk, max(c_ns.shape[1], 1))
    vs_exc = c_ns.topk(k_ns, dim=-1).values.sum(dim=-1) / total_ns
    return [float(vs_exc[h]) for h in heads]


# ── main ─────────────────────────────────────────────────────────────────────

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lengths = [int(x) for x in args.lengths.split()]
    preset = args.nope_preset
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    try:
        tok = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)
        _ = tok(["x"])
    except Exception:
        tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    config = AutoConfig.from_pretrained(args.checkpoint)
    config.attn_implementation = "eager"
    config.pe_method = "rotary"
    config.rope_theta = args.rope_theta
    n_layer = int(getattr(config, "n_layer", 12))
    n_head = int(getattr(config, "n_head", 12))
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, config=config, torch_dtype=torch.float32, trust_remote_code=True)
    model.eval().to(device)

    # choose head set
    vert = set(VERTICAL_HEADS)
    if preset == "none":
        head_set = set()
    elif preset == "vertical":
        head_set = set(vert)
    elif preset == "all":
        head_set = {(l, h) for l in range(n_layer) for h in range(n_head)}
    elif preset == "random":
        non_vert = [(l, h) for l in range(n_layer) for h in range(n_head)
                    if (l, h) not in vert]
        idx = rng.choice(len(non_vert), size=len(vert), replace=False)
        head_set = {non_vert[i] for i in idx}
    else:
        raise ValueError(preset)
    n_patched = apply_nope(model, head_set, n_layer, n_head) if head_set else 0
    print(f"preset={preset} | RoPE switched off on {len(head_set)} heads "
          f"across {n_patched} layers", flush=True)
    vert_by_layer = collections.defaultdict(list)
    for (l, h) in VERTICAL_HEADS:
        vert_by_layer[l].append(h)

    # load all cases once
    all_cases = []
    with open(args.jsonl) as f:
        for line in f:
            if line.strip():
                all_cases.append(json.loads(line))

    rows = []
    for L in lengths:
        cases = [c for c in all_cases
                 if int(c.get("meta", {}).get("target_total_tokens", -1)) == L][:args.n_case]
        print(f"[L={L}] {len(cases)} cases", flush=True)
        for ci, rec in enumerate(cases):
            prompt_str = render_prompt(rec)
            answer_str = f" {rec['answer']}"
            p_ids = tok(prompt_str, add_special_tokens=False)["input_ids"]
            a_ids = tok(answer_str, add_special_tokens=False)["input_ids"]
            # NO truncation to L: cases are already selected by target_total_tokens==L,
            # so the natural length ~= L. Truncating to L cut off the trailing "Answer:"
            # for cases whose actual length slightly exceeds L (matches canonical
            # eval_hotpot_long.py which uses truncation=False).
            full_ids = p_ids + a_ids
            n_ans = len(a_ids)
            ids = torch.tensor([full_ids], dtype=torch.long, device=device)

            # forward for PPL (+ VS unless fast path)
            with torch.no_grad():
                out = model(ids, output_attentions=not args.no_attn)
            logits = out.logits[0].float()             # [T, V]
            lsm = torch.log_softmax(logits[:-1], dim=-1)
            tgt = ids[0, 1:]
            nll = -lsm[torch.arange(tgt.shape[0]), tgt]  # [T-1]
            ppl_full = float(nll.mean())
            ppl_answer = float(nll[-n_ans:].mean()) if n_ans > 0 else float("nan")
            if args.no_attn:
                vs_vertical = float("nan")
            else:
                vs_vals = []
                for l, heads in vert_by_layer.items():
                    vs_vals.extend(vs_exc_for_heads(out.attentions[l][0], heads))
                vs_vertical = float(np.mean(vs_vals))
            del out
            if device.type == "cuda":
                torch.cuda.empty_cache()

            # generation for F1/EM — full prompt (keeps trailing "Answer:")
            prompt_ids = torch.tensor([p_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                gen = model.generate(prompt_ids, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, pad_token_id=tok.pad_token_id)
            pred = tok.decode(gen[0, prompt_ids.shape[1]:],
                              skip_special_tokens=True).strip().split("\n")[0].strip()
            f1, em = best_f1_em(pred, rec.get("answer_aliases", [rec.get("answer", "")]))

            rows.append({"length": L, "condition": preset, "case_id": ci,
                         "actual_len": len(full_ids), "prompt_len": min(len(p_ids), L),
                         "ppl_full": ppl_full, "ppl_answer": ppl_answer,
                         "vs_vertical": vs_vertical, "f1": f1, "em": em})
            if (ci + 1) % 10 == 0 or ci + 1 == len(cases):
                print(f"  [L={L}] {ci+1}/{len(cases)} | ppl_full={ppl_full:.3f} "
                      f"vs={vs_vertical:.3f} f1={f1:.2f}", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    path = out_dir / f"nope_eval_rows_{preset}.parquet"
    df.to_parquet(path, index=False)
    summ = (df.groupby("length")[["ppl_full", "ppl_answer", "vs_vertical", "f1", "em"]]
            .mean().round(4))
    print(f"\nDone preset={preset} -> {path}\n{summ.to_string()}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="PAT-203 NoPE-on-vertical-head extrapolation eval")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--nope_preset", required=True,
                    choices=["none", "vertical", "random", "all"])
    ap.add_argument("--lengths", default="512 2048 4096")
    ap.add_argument("--n_case", type=int, default=80)
    ap.add_argument("--max_new_tokens", type=int, default=24)
    ap.add_argument("--rope_theta", type=float, default=10000.0)
    ap.add_argument("--no_attn", action="store_true",
                    help="fast path: skip output_attentions/VS (PPL+F1 only)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()

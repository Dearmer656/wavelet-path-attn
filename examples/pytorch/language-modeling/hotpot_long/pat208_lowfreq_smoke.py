#!/usr/bin/env python3
"""pat208_lowfreq_smoke.py — verify the low-freq (long-period) NoPE dim-mask.

Checks:
 [1] lowfreq_dim_mask selects exactly the pairs with P_m>cutoff; for theta=1e4,
     D=64, cutoff=512 -> 16 pairs (m=16..31) -> dims 32..63.
 [2] dim-masked rotate is IDENTITY on masked dims and EQUAL to standard RoPE on
     the unmasked (high-freq) dims (max abs diff 0 on each side).
 [3] a real forward runs and lowfreq logits sit BETWEEN none (full RoPE) and all
     (full NoPE) — sanity that it's a partial intervention.
"""
import argparse, json, numpy as np, torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from eval_nope_vertical import (apply_lowfreq_nope, lowfreq_dim_mask,
                                make_dim_masked_rotate)


def run(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = AutoConfig.from_pretrained(a.checkpoint)
    cfg.attn_implementation = "eager"; cfg.pe_method = "rotary"
    nh = int(cfg.n_head); D = int(cfg.n_embd // nh)
    model = AutoModelForCausalLM.from_pretrained(
        a.checkpoint, config=cfg, torch_dtype=torch.float32, trust_remote_code=True).eval().to(dev)

    # locate one rotary_emb
    re_emb = None
    for _, m in model.named_modules():
        if hasattr(m, "rotary_emb"):
            re_emb = m.rotary_emb; break
    freqs = re_emb.freqs

    # [1] mask selection
    mask, n_off, periods = lowfreq_dim_mask(freqs, D, a.cutoff)
    off_pairs = [i for i in range(D // 2) if periods[i] > a.cutoff]
    print(f"[1] head_dim={D} n_pairs={D//2} cutoff={a.cutoff}")
    print(f"    periods: min={periods.min():.1f} max={periods.max():.1f}")
    print(f"    pairs with P>cutoff: {n_off} -> first off pair m={off_pairs[0]} "
          f"(P={periods[off_pairs[0]]:.1f}); last on pair m={off_pairs[0]-1} "
          f"(P={periods[off_pairs[0]-1]:.1f})")
    off_dims = [d for d in range(D) if mask[d] > 0]
    print(f"    masked dims: {off_dims[0]}..{off_dims[-1]} (count {len(off_dims)})")
    assert n_off == sum(1 for p in periods if p > a.cutoff)

    # [2] identity on masked dims, standard on unmasked
    t = torch.randn(1, nh, 40, D, device=dev)
    orig = re_emb.rotate_queries_or_keys
    wrapped = make_dim_masked_rotate(orig, mask)
    rot_std = orig(t)
    rot_lf = wrapped(t)
    md = mask.to(dev).bool()
    diff_masked = (rot_lf[..., md] - t[..., md]).abs().max().item()       # should be 0
    diff_unmask = (rot_lf[..., ~md] - rot_std[..., ~md]).abs().max().item()  # should be 0
    print(f"[2] masked-dim |lowfreq - identity| = {diff_masked:.3e} (expect 0)")
    print(f"    unmask-dim |lowfreq - stdRoPE| = {diff_unmask:.3e} (expect 0)")
    assert diff_masked < 1e-6 and diff_unmask < 1e-6

    # [3] real forward: none vs lowfreq vs all (full NoPE)
    try:
        tok = AutoTokenizer.from_pretrained(a.checkpoint, use_fast=True); tok(["x"])
    except Exception:
        tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    rec = None
    for line in open(a.jsonl):
        if line.strip():
            r = json.loads(line)
            if int(r.get("meta", {}).get("target_total_tokens", -1)) == a.L:
                rec = r; break
    lines = [f"{ti}: {s}" for ti, sents in rec["context"] for s in sents]
    ids = tok(f"Context:\n{chr(10).join(lines)}\nQuestion: {rec['question']}\nAnswer:",
              add_special_tokens=False)["input_ids"][:a.L]
    x = torch.tensor([ids], device=dev)
    with torch.no_grad():
        lg_none = model(x).logits[0, -1].float().cpu()
    n_lay, n_off2, _ = apply_lowfreq_nope(model, list(range(int(cfg.n_layer))), a.cutoff,
                                          int(cfg.n_layer), nh)
    with torch.no_grad():
        lg_lf = model(x).logits[0, -1].float().cpu()
    d_lf = (lg_lf - lg_none).abs().max().item()
    print(f"[3] applied low-freq NoPE on all {n_lay} layers ({n_off2} pairs/layer); "
          f"forward OK; |logit_lowfreq - logit_none|_max = {d_lf:.3f} (expect >0, finite)")
    assert np.isfinite(d_lf) and d_lf > 0
    print("ALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--cutoff", type=float, default=512.0)
    ap.add_argument("--L", type=int, default=2048)
    run(ap.parse_args())

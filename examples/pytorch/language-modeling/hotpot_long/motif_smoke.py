#!/usr/bin/env python3
"""motif_smoke.py — validate the PAT-217 substitution mechanism end-to-end (correctness).

Checks (one L2048 case):
 [1] forward runs after substitution;
 [2] a NON-selected head's attention is UNCHANGED vs clean (only selected heads touched);
 [3] a selected (broken) head's OOD attention mass (offset>512) is REDUCED vs clean RoPE
     (the splash is suppressed by NoPE+motif substitution).
"""
import argparse, json
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from motif_substitution import apply_motif_substitution, select_broken_heads


def render_trained(ex):
    lines = [f"{t}: {s}" for t, sents in ex["context"] for s in sents]
    return f"Context:\n{chr(10).join(lines)}\nQuestion: {ex['question']}\nAnswer:"


def ood_mass(A, tr=512):
    # A [T,T] post-softmax; mean over rows i>=tr of attention mass at offset>tr (j < i-tr)
    T = A.shape[0]; vals = []
    for i in range(tr, T):
        vals.append(A[i, :max(0, i - tr)].sum())
    return float(np.mean(vals)) if vals else 0.0


def run(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(a.checkpoint)
    cfg.attn_implementation = "eager"; cfg.pe_method = "rotary"
    model = AutoModelForCausalLM.from_pretrained(
        a.checkpoint, config=cfg, torch_dtype=torch.float32, trust_remote_code=True).eval().to(dev)

    rec = None
    for line in open(a.jsonl):
        if line.strip():
            r = json.loads(line)
            if int(r.get("meta", {}).get("target_total_tokens", -1)) == a.L:
                rec = r; break
    ids = tok(render_trained(rec), add_special_tokens=False)["input_ids"][:a.L]
    x = torch.tensor([ids], device=dev)

    broken = select_broken_heads(a.recon_csv, a.top_k)
    sel = set(broken)
    print(f"selected {len(broken)} broken heads; top5={broken[:5]}", flush=True)
    good = next((l, h) for l in range(2) for h in range(12) if (l, h) not in sel)
    bad = broken[0]

    with torch.no_grad():
        att0 = model(x, output_attentions=True).attentions
    A0_good = att0[good[0]][0, good[1]].float().cpu().numpy()
    A0_bad = att0[bad[0]][0, bad[1]].float().cpu().numpy()

    apply_motif_substitution(model, a.npz, broken, lam=a.lam, mode="real")
    with torch.no_grad():
        att1 = model(x, output_attentions=True).attentions
    A1_good = att1[good[0]][0, good[1]].float().cpu().numpy()
    A1_bad = att1[bad[0]][0, bad[1]].float().cpu().numpy()

    d_good = np.abs(A1_good - A0_good).max()
    om0, om1 = ood_mass(A0_bad), ood_mass(A1_bad)
    print(f"[2] non-selected head L{good[0]}H{good[1]}: max|Δattn| = {d_good:.2e} (expect ~0)")
    print(f"[3] selected head L{bad[0]}H{bad[1]}: OOD mass  clean={om0:.3f} -> subst={om1:.3f} "
          f"({'REDUCED' if om1 < om0 else 'NOT reduced'})")
    ok = (d_good < 1e-4) and (om1 <= om0 + 1e-3)
    print("SMOKE", "PASS" if ok else "CHECK", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--recon_csv", required=True)
    ap.add_argument("--top_k", type=int, default=16)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--L", type=int, default=2048)
    run(ap.parse_args())

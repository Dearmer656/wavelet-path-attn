#!/usr/bin/env python3
"""distill_bias_lowfreq.py — PAT-217 × PAT-208: LOW-FREQ residual structural motif.

Motivation (combine the two findings):
  * PAT-208: switching RoPE off ONLY on the low-freq (long-period P_m>train_len) dims
    ~solves extrapolation (finetune all-layer low-freq NoPE: L2048 0.741 / L4096 0.698).
    The OOD splash is carried specifically by those period-underexposed dims.
  * PAT-217: the training-length attention structure factorizes as slash s(δ)+sink v(j),
    and injecting that frozen motif on NoPE'd heads adds value.

This script distills the motif for the LOW-FREQ subspace ONLY, and — crucially — as a
RESIDUAL, so it stacks cleanly on top of PAT-208's low-freq NoPE without double-counting
content. At test the low-freq dims keep live NoPE content-QK (B_nope_lf); RoPE at training
added the positional structure B_rope_lf. We distill

      residual_h(i,j) = B_rope_lf,h(i,j) − B_nope_lf,h(i,j)          (content-averaged, L512)

= exactly the position-dependent part RoPE contributes on the low-freq dims — then fit the
length-extendable additive model residual_h(i,j) ≈ s_h(δ=i−j) + v_h(j). At extrapolation the
low-freq dims are NoPE (live content, no OOD recurrence) + this frozen residual (extended via
hold_last for slash, median for sink). High-freq dims stay on standard RoPE (one matmul).

Two matmuls appear ONLY here (distill): B_rope_lf and B_nope_lf on the low-freq dims to form
the residual. At test it stays a single QK matmul (dim-masked rotation).

Saves distilled_bias_lowfreq_L{L}.npz with the SAME schema as distill_bias_step1.py
(s[144,L], v[144,L], var_explained, slash_frac) so the downstream substitution path is reused.
"""
import argparse, json, os, types
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

_POST = {}   # _POST[layer] = [(q_pre,q_post), (k_pre,k_post)]


def render_trained(ex):
    lines = [f"{t}: {s}" for t, sents in ex["context"] for s in sents]
    return f"Context:\n{chr(10).join(lines)}\nQuestion: {ex['question']}\nAnswer:"


def install_prepost_capture(model):
    """Capture BOTH the pre-rotation input t (NoPE q/k) and post-rotation output (RoPE q/k)."""
    for _, mod in model.named_modules():
        re = getattr(mod, "rotary_emb", None)
        if re is None:
            continue
        li = int(getattr(mod, "layer_idx", 0))
        orig = re.rotate_queries_or_keys

        def wrapped(self, t, *a, _li=li, _orig=orig, **kw):
            out = _orig(t, *a, **kw)
            _POST.setdefault(_li, []).append((t.detach().float(), out.detach().float()))
            return out
        re.rotate_queries_or_keys = types.MethodType(wrapped, re)


def lowfreq_dim_mask(freqs, head_dim, period_cutoff):
    """[head_dim] mask = 1.0 on dims of 2D pairs whose RoPE period P_m=2*pi/theta_m > cutoff.
    Interleaved convention pair m -> dims (2m, 2m+1). Returns (mask, n_pairs_off, periods)."""
    inv = freqs.detach().float().cpu().numpy()
    periods = 2.0 * np.pi / np.clip(inv, 1e-12, None)
    mask = np.zeros(head_dim, np.float32)
    n_pairs = head_dim // 2
    n_off = 0
    for mi in range(min(n_pairs, len(periods))):
        if periods[mi] > period_cutoff:
            mask[2 * mi] = 1.0; mask[2 * mi + 1] = 1.0
            n_off += 1
    return mask, n_off, periods


def additive_fit(B, ii, jj, off, cnt_off, cnt_col, T, iters=8):
    """B[T,T] lower-tri; fit B[i,j] ~ s[i-j] + v[j]. Returns s,v,var_expl,slash_e,sink_e."""
    b = B[ii, jj]
    s = np.zeros(T); v = np.zeros(T)
    for _ in range(iters):
        s = np.bincount(off, weights=(b - v[jj]), minlength=T) / cnt_off
        v = np.bincount(jj, weights=(b - s[off]), minlength=T) / cnt_col
        v -= v.mean()
    recon = s[off] + v[jj]
    ss_res = float(((b - recon) ** 2).sum())
    ss_tot = float(((b - b.mean()) ** 2).sum()) + 1e-9
    return s, v, 1.0 - ss_res / ss_tot, float((s[off] ** 2).sum()), float((v[jj] ** 2).sum())


def run(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out, exist_ok=True)
    try:
        tok = AutoTokenizer.from_pretrained(a.checkpoint, use_fast=True); tok(["x"])
    except Exception:
        tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(a.checkpoint)
    cfg.attn_implementation = "eager"; cfg.pe_method = "rotary"
    model = AutoModelForCausalLM.from_pretrained(
        a.checkpoint, config=cfg, torch_dtype=torch.float32, trust_remote_code=True).eval().to(dev)
    install_prepost_capture(model)
    nl, nh, T = int(cfg.n_layer), int(cfg.n_head), a.L
    D = cfg.n_embd // nh
    scale = 1.0 / np.sqrt(D)

    # low-freq dim mask (shared across layers for standard RoPE)
    freqs = None
    for _, mod in model.named_modules():
        re = getattr(mod, "rotary_emb", None)
        if re is not None and hasattr(re, "freqs"):
            freqs = re.freqs; break
    assert freqs is not None, "no rotary_emb.freqs found"
    dmask_np, n_off, periods = lowfreq_dim_mask(freqs, D, a.period_cutoff)
    print(f"low-freq (P>{a.period_cutoff}) NoPE dims = {int(dmask_np.sum())}/{D} "
          f"({n_off}/{len(periods)} pairs)", flush=True)
    dmask = torch.tensor(dmask_np, device=dev)   # [D], 1.0 on low-freq dims

    cases = []
    for line in open(a.jsonl):
        if not line.strip():
            continue
        r = json.loads(line)
        if int(r.get("meta", {}).get("target_total_tokens", -1)) != a.L:
            continue
        ids = tok(render_trained(r), add_special_tokens=False)["input_ids"]
        if len(ids) >= a.L:
            cases.append(ids[:a.L])
        if len(cases) >= a.n_cases:
            break
    print(f"L{a.L}: {len(cases)} cases", flush=True)

    # accumulate content-averaged LOW-FREQ residual template = B_rope_lf - B_nope_lf
    Bsum = torch.zeros(nl, nh, T, T)
    dm = dmask.view(1, 1, D)                       # broadcast over [nh,T,D]
    for ci, ids in enumerate(cases):
        _POST.clear()
        with torch.no_grad():
            model(torch.tensor([ids], device=dev))
        for l in range(nl):
            (q_pre, q_post) = _POST[l][0]           # each [1,nh,T,D]
            (k_pre, k_post) = _POST[l][1]
            qp = (q_post[0] * dm); kp = (k_post[0] * dm)     # low-freq post-RoPE
            qn = (q_pre[0] * dm);  kn = (k_pre[0] * dm)      # low-freq pre-RoPE (NoPE)
            B_rope = torch.einsum("htd,hsd->hts", qp, kp) * scale
            B_nope = torch.einsum("htd,hsd->hts", qn, kn) * scale
            Bsum[l] += (B_rope - B_nope).cpu()      # RESIDUAL = positional structure RoPE adds
        print(f"  case {ci} done", flush=True)
    B = (Bsum / len(cases)).numpy()                 # [nl,nh,T,T] low-freq residual

    tril = np.tril_indices(T); ii, jj = tril; off = ii - jj
    cnt_off = np.maximum(np.bincount(off, minlength=T), 1)
    cnt_col = np.maximum(np.bincount(jj, minlength=T), 1)

    S_tab = np.zeros((nl * nh, T)); V_tab = np.zeros((nl * nh, T))
    rows = []
    for l in range(nl):
        for h in range(nh):
            s, v, ve, sl, sk = additive_fit(B[l, h], ii, jj, off, cnt_off, cnt_col, T)
            idx = l * nh + h
            S_tab[idx] = s; V_tab[idx] = v
            tot = sl + sk + 1e-9
            rows.append((l, h, ve, sl / tot, sk / tot))
    ve_all = np.array([r[2] for r in rows]); slfrac = np.array([r[3] for r in rows])
    print(f"\nlow-freq residual slash+sink fit over {nl*nh} heads:")
    print(f"  var-explained: mean={ve_all.mean():.3f}  median={np.median(ve_all):.3f}  "
          f"min={ve_all.min():.3f}")
    print(f"  slash share: mean={slfrac.mean():.3f}  (sink share mean={1-slfrac.mean():.3f})")

    out_npz = os.path.join(a.out, f"distilled_bias_lowfreq_L{a.L}.npz")
    np.savez(out_npz, s=S_tab, v=V_tab, n_layer=nl, n_head=nh, L=T,
             var_explained=ve_all, slash_frac=slfrac,
             period_cutoff=float(a.period_cutoff), lowfreq_dim_mask=dmask_np)
    print(f"saved -> {out_npz}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--L", type=int, default=512)
    ap.add_argument("--n_cases", type=int, default=20)
    ap.add_argument("--period_cutoff", type=float, default=512.0)
    ap.add_argument("--out", required=True)
    run(ap.parse_args())

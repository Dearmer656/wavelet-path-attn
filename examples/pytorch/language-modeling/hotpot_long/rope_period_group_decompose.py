#!/usr/bin/env python3
"""rope_period_group_decompose.py — PAT-209 step-1 KILL-GATE.

Motivated by PAT-208: the OOD slash/splash collapse comes from the preserved in-range
cone template g(δ)=uᵀR(δ)v = Σ_m A_m cos(δθ_m+φ_m) producing uncontrolled large-offset
recurrence spikes at δ>L_train. PAT-209 proposes dim-specific (per-RoPE-pair) position
exposure so each frequency sees ≥1 full period during training. That only helps IF the
OOD spikes are dominated by *period-underexposed* frequencies (period P_m>L_train).

This script DECOMPOSES the OOD spikes by period group, with NO training:

  g_short(δ) = Σ_{P_m ≤ L_train} A_m cos(δθ_m+φ_m)   (frequencies that saw ≥1 full cycle)
  g_long (δ) = Σ_{P_m >  L_train} A_m cos(δθ_m+φ_m)   (period-underexposed frequencies)

Per head we report, over the OOD offset range δ>L_train:
  - max|g_full|, max|g_long|, max|g_short|
  - at δ* = argmax_{δ>L} g_full: the long-group share  g_long(δ*) / (|g_long|+|g_short|)(δ*)
  - long-group amplitude energy share  Σ_{long}A_m² / Σ_m A_m²
  - corr(g_full, actual post-RoPE logit) in-range vs OOD  (proxy-quality check)

Self-validation: g computed analytically from re.freqs (per-pair formula) is asserted to
match g computed by feeding u,v THROUGH rotary_emb (so the interleaved-pair convention and
the frequency values are both verified before any conclusion is drawn).

GATE: if OOD spikes are NOT dominated by the long-period (P_m>L_train) group → the
dim-specific exposure idea is deprioritized (it cannot fix what it doesn't cause).
"""
import argparse, json, types, os
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# heads spanning the late slash/hijack layers (8-11) + two mid-layer references
HEADS = [(4, 7), (5, 3), (8, 8), (9, 5), (10, 9), (11, 3)]
_PRE, _POST = {}, {}


def render_trained(ex):
    lines = [f"{t}: {s}" for t, sents in ex["context"] for s in sents]
    return f"Context:\n{chr(10).join(lines)}\nQuestion: {ex['question']}\nAnswer:"


def install(model):
    for _, mod in model.named_modules():
        re = getattr(mod, "rotary_emb", None)
        if re is None:
            continue
        li = int(getattr(mod, "layer_idx", 0))
        orig = re.rotate_queries_or_keys

        def wrapped(self, t, *args, _li=li, _orig=orig, **kw):
            _PRE.setdefault(_li, []).append(t.detach().float().cpu())
            out = _orig(t, *args, **kw)
            _POST.setdefault(_li, []).append(out.detach().float().cpu())
            return out
        re.rotate_queries_or_keys = types.MethodType(wrapped, re)


def dom_dir(M):
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    u = Vt[0]
    if float(np.sum(M @ u)) < 0:
        u = -u
    return u


def pair_amp_phase(u, v, theta):
    """g(δ)=Σ_m A_m cos(δθ_m+φ_m) for interleaved pairs (2m,2m+1), std 2D rotation.
    C_m=u_{2m}v_{2m}+u_{2m+1}v_{2m+1}, S_m=u_{2m+1}v_{2m}-u_{2m}v_{2m+1}.
    g_pair(δ)=C_m cos(δθ_m)+S_m sin(δθ_m)=A_m cos(δθ_m-φ_m), A=√(C²+S²), φ=atan2(S,C)."""
    u0, u1 = u[0::2], u[1::2]
    v0, v1 = v[0::2], v[1::2]
    C = u0 * v0 + u1 * v1
    S = u1 * v0 - u0 * v1
    A = np.sqrt(C * C + S * S)
    # actual logit uses the relative encoding qᵀR(i)ᵀR(j)k = qᵀR(j-i)k = uᵀR(-δ)v,
    # so g_pair(δ)=C cos(δθ) - S sin(δθ) = A cos(δθ+φ), φ=atan2(S,C).
    phi = np.arctan2(S, C)
    return A, phi, theta


def g_of_delta(deltas, A, phi, theta, mask=None):
    sel = np.ones_like(A, dtype=bool) if mask is None else mask
    # [n_pairs_sel, n_delta]
    arg = np.outer(theta[sel], deltas)            # δθ
    contrib = A[sel][:, None] * np.cos(arg + phi[sel][:, None])
    return contrib.sum(axis=0)


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
    install(model)
    cases = []
    for line in open(a.jsonl):
        if not line.strip():
            continue
        r = json.loads(line)
        if int(r.get("meta", {}).get("target_total_tokens", -1)) == a.L:
            cases.append(r)
        if len(cases) > a.case_idx + 64:
            break
    sel = None
    for r in cases[a.case_idx:]:
        ids = tok(render_trained(r), add_special_tokens=False)["input_ids"]
        if len(ids) >= a.L:
            sel = ids[:a.L]; break
    with torch.no_grad():
        model(torch.tensor([sel], device=dev))

    T, tr = a.L, a.train
    D = model.config.n_embd // model.config.n_head
    scale = 1.0 / np.sqrt(D)
    i0 = T - 1
    deltas = np.arange(1, T)                       # δ = 1 .. T-1
    ood = deltas > tr
    blocks = list(model.transformer.h)

    rows = []
    fig, axes = plt.subplots(len(HEADS), 1, figsize=(15, 3.0 * len(HEADS)), squeeze=False)
    print(f"{'head':>7} | n_long | E_long% | maxOOD g_full | g_long(δ*)% | corr_in | corr_ood", flush=True)
    for ridx, (l, h) in enumerate(HEADS):
        qpre = _PRE[l][0][0, h].numpy(); kpre = _PRE[l][1][0, h].numpy()
        u = dom_dir(qpre[:tr]); v = dom_dir(kpre[:tr])      # in-range cone (unit), len D

        re = blocks[l].attn.rotary_emb
        theta = re.freqs.detach().float().cpu().numpy().reshape(-1)   # inv_freq, len D/2
        assert theta.shape[0] == D // 2, (theta.shape, D)
        P = 2 * np.pi / theta                                # period per pair
        long_mask = P > tr                                   # period-underexposed

        A, phi, th = pair_amp_phase(u, v, theta)
        g_full = g_of_delta(deltas, A, phi, th) * scale
        g_long = g_of_delta(deltas, A, phi, th, long_mask) * scale
        g_short = g_of_delta(deltas, A, phi, th, ~long_mask) * scale

        # --- self-validation: feed unit cone through rotary_emb, compare g_full ---
        re.interpolate_factor = 1.0; re.cache_if_possible = False
        re.cached_freqs_seq_len = 0; re.cached_scales_seq_len = 0
        with torch.no_grad():
            U = torch.tensor(u, dtype=torch.float32, device=dev).view(1, 1, 1, D).expand(1, 1, T, D).contiguous()
            V = torch.tensor(v, dtype=torch.float32, device=dev).view(1, 1, 1, D).expand(1, 1, T, D).contiguous()
            Ru = re.rotate_queries_or_keys(U)[0, 0].float().cpu().numpy()
            Rv = re.rotate_queries_or_keys(V)[0, 0].float().cpu().numpy()
        g_feed_row = (Ru[i0] @ Rv.T) * scale                 # indexed by key j; δ=i0-j
        g_feed = g_feed_row[i0 - deltas]                     # align to δ grid
        vmax_err = float(np.max(np.abs(g_feed - g_full)))
        assert vmax_err < 1e-3, f"L{l}H{h} analytic vs feed-through mismatch {vmax_err}"

        # actual post-RoPE logit profile for query i0
        qpo = _POST[l][0][0, h].numpy(); kpo = _POST[l][1][0, h].numpy()
        S_row = (qpo[i0] @ kpo.T) * scale
        actual = S_row[i0 - deltas]

        def corr(m):
            if m.sum() < 3:
                return float("nan")
            return float(np.corrcoef(g_full[m], actual[m])[0, 1])
        corr_in = corr((deltas >= 1) & (deltas < tr))
        corr_ood = corr(ood)

        # OOD spike attribution
        dstar = deltas[ood][np.argmax(g_full[ood])]
        j = dstar - 1                                        # index into deltas array
        gl_star = abs(g_long[ood][np.argmax(g_full[ood])])
        gs_star = abs(g_short[ood][np.argmax(g_full[ood])])
        long_share_star = gl_star / (gl_star + gs_star + 1e-9)
        maxood_full = float(np.max(np.abs(g_full[ood])))
        maxood_long = float(np.max(np.abs(g_long[ood])))
        maxood_short = float(np.max(np.abs(g_short[ood])))
        E_long = float(np.sum(A[long_mask] ** 2) / (np.sum(A ** 2) + 1e-12))
        # OOD-region rms share (how much "spiking energy" each group carries beyond train)
        rms_long = float(np.sqrt(np.mean(g_long[ood] ** 2)))
        rms_short = float(np.sqrt(np.mean(g_short[ood] ** 2)))
        rms_long_share = rms_long / (rms_long + rms_short + 1e-9)

        rows.append(dict(layer=l, head=h, n_pairs=int(D // 2), n_long=int(long_mask.sum()),
                         E_long_share=E_long, maxood_full=maxood_full, maxood_long=maxood_long,
                         maxood_short=maxood_short, dstar=int(dstar),
                         long_share_at_peak=long_share_star, rms_long_share=rms_long_share,
                         corr_in=corr_in, corr_ood=corr_ood, selfval_err=vmax_err))
        print(f"L{l}H{h} | {int(long_mask.sum()):>6d} | {100*E_long:6.1f} | {maxood_full:13.2f} | "
              f"{100*long_share_star:10.1f} | {corr_in:+.3f} | {corr_ood:+.3f}", flush=True)

        ax = axes[ridx][0]
        ax.plot(deltas, actual, lw=0.3, color="0.8", label="actual post-RoPE logit (right axis)")
        ax.axvline(tr, color="k", ls="--", lw=0.9)
        ax.set_xlabel("offset δ = i - j"); ax.set_ylabel("actual logit", color="0.6")
        ax2 = ax.twinx()       # templates are unit-cone scale (~0.1); separate axis
        ax2.plot(deltas, g_full, lw=0.7, color="k", label="g_full (rank-1 cone template)")
        ax2.plot(deltas, g_short, lw=0.7, color="tab:blue", label=f"g_short (P≤{tr}, {int((~long_mask).sum())} pairs)")
        ax2.plot(deltas, g_long, lw=0.7, color="tab:red", label=f"g_long (P>{tr}, {int(long_mask.sum())} pairs)")
        ax2.set_ylabel("cone template g(δ)")
        ax.set_title(f"L{l}H{h}: OOD spike attribution — long-group share at peak {100*long_share_star:.0f}%, "
                     f"E_long {100*E_long:.0f}%, corr_ood {corr_ood:+.2f}", fontsize=9)
        if ridx == 0:
            ax2.legend(fontsize=7, ncol=3, loc="upper left")
    fig.suptitle(f"PAT-209 step-1: period-group decomposition of OOD spikes (case {a.case_idx}, L{a.L}, train={tr})", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "rope_period_group_decompose.png"), dpi=130, bbox_inches="tight")

    import csv
    with open(os.path.join(a.out, "period_group_decompose.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    ls_peak = np.array([r["long_share_at_peak"] for r in rows])
    ls_rms = np.array([r["rms_long_share"] for r in rows])
    el = np.array([r["E_long_share"] for r in rows])
    print("\n=== PAT-209 STEP-1 GATE SUMMARY ===", flush=True)
    print(f"heads={len(rows)}  median long-share at OOD peak = {np.median(ls_peak):.2f}  "
          f"median OOD rms long-share = {np.median(ls_rms):.2f}  median E_long_share = {np.median(el):.2f}", flush=True)
    verdict = "PASS (long-period group dominates OOD spikes → dim-specific exposure targets the cause)" \
        if np.median(ls_peak) >= 0.5 else \
        "FAIL (OOD spikes dominated by SHORT-period group → dim-specific exposure deprioritized)"
    print(f"VERDICT: {verdict}", flush=True)
    print(f"saved -> {a.out}/", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--L", type=int, default=2048)
    ap.add_argument("--train", type=int, default=512)
    ap.add_argument("--case_idx", type=int, default=0)
    ap.add_argument("--out", required=True)
    run(ap.parse_args())

#!/usr/bin/env python3
"""offset_motif_smoke.py — Gate-1 for the flexible-motif direction.

Hypothesis: a length-agnostic, offset-space (δ = i-j) decomposition of the attention
PATTERN reveals a "length-emergent" motif (the splash / displaced-peak) that the
hand-picked Vertical/Diagonal/Periodicity (V/D/P) vocabulary cannot describe, and that
marks the extrapolation-broken heads.

Method (per length L, rotary ckpt, N cases):
  1. Forward with output_attentions; for each (layer,head) build the offset profile
       g_h(δ) = mean_i A[i, i-δ]      (case-averaged)   -> length-agnostic curve.
  2. NMF-decompose the stack G=[144 heads x L] into R motifs m_k(δ) + usage U[144,R].
  3. Identify the "splash" motif = component with the most mass at large δ.
  4. Hand scores per head: V (sink/prefix mass), D (near-diagonal mass), P (periodicity
     via FFT of g).  GATE-1: can [V,D,P] linearly predict splash-motif usage?  Low R^2
     => the flexible basis captures a motif OUTSIDE the V/D/P vocabulary.
Outputs: motif curves + G heatmap + usage-vs-handscore scatter, CSV, printed verdict.
"""
import argparse, json, os
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import NMF
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render_trained(ex):
    lines = [f"{t}: {s}" for t, sents in ex["context"] for s in sents]
    return f"Context:\n{chr(10).join(lines)}\nQuestion: {ex['question']}\nAnswer:"


def select_cases(tok, jsonl, L, n):
    out = []
    for line in open(jsonl):
        if not line.strip():
            continue
        r = json.loads(line)
        if int(r.get("meta", {}).get("target_total_tokens", -1)) != L:
            continue
        ids = tok(render_trained(r), add_special_tokens=False)["input_ids"]
        if len(ids) >= L:
            out.append(ids[:L])
        if len(out) >= n:
            break
    return out


def offset_profiles(att, ii, jj, offs, counts, T, accum):
    # att: tuple[nl] of [1,nh,T,T]; accum[l,h] += g_h(δ)
    for l, A in enumerate(att):
        A = A[0]  # [nh,T,T]
        vals = A[:, ii, jj]                     # [nh, num_lower]
        for h in range(A.shape[0]):
            sums = torch.zeros(T, device=A.device).scatter_add_(0, offs, vals[h])
            accum[l][h] += (sums / counts).cpu().numpy()


def hand_scores(g, T):
    # g: [L] offset profile (mean attention per δ), non-negative
    s = g.sum() + 1e-9
    D = g[:8].sum() / s                                   # near-diagonal / local
    # periodicity: FFT power of g over δ>=16, excluding DC, peak / total
    seg = g[16:]
    if seg.size > 8:
        f = np.abs(np.fft.rfft(seg - seg.mean()))
        P = (f[1:].max() / (f.sum() + 1e-9))
    else:
        P = 0.0
    return D, P


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
    nl, nh, T = int(cfg.n_layer), int(cfg.n_head), a.L

    cases = select_cases(tok, a.jsonl, a.L, a.n_cases)
    print(f"L{a.L}: {len(cases)} cases", flush=True)

    ii, jj = torch.tril_indices(T, T, device=dev)          # j<=i (causal)
    offs = (ii - jj).to(dev)                                # δ per lower-tri entry
    counts = torch.bincount(offs, minlength=T).float().clamp(min=1)
    accum = [[np.zeros(T) for _ in range(nh)] for _ in range(nl)]
    # also accumulate a prefix/sink (vertical) score per head: mass on cols<4
    sink = np.zeros((nl, nh))

    for ci, ids in enumerate(cases):
        with torch.no_grad():
            out = model(torch.tensor([ids], device=dev), output_attentions=True)
        att = out.attentions
        offset_profiles(att, ii, jj, offs, counts, T, accum)
        for l, A in enumerate(att):
            sink[l] += A[0, :, :, :4].sum(-1).mean(-1).cpu().numpy()   # mean_i mass on first 4 keys
        del out, att
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  case {ci} done", flush=True)

    G = np.stack([accum[l][h] / len(cases) for l in range(nl) for h in range(nh)])  # [144, T]
    sink = (sink / len(cases)).reshape(-1)                                          # [144] V score
    Dsc = np.zeros(len(G)); Psc = np.zeros(len(G))
    for k in range(len(G)):
        Dsc[k], Psc[k] = hand_scores(G[k], T)

    # ---- NMF over offset profiles ----
    R = a.R
    Gpos = np.clip(G, 0, None)
    nmf = NMF(n_components=R, init="nndsvda", max_iter=800, random_state=0)
    U = nmf.fit_transform(Gpos)      # [144, R]
    M = nmf.components_              # [R, T] motif curves
    # identify splash motif = component with most mass at δ>256
    far = M[:, 256:].sum(1) / (M.sum(1) + 1e-9)
    peakd = M.argmax(1)
    splash_k = int(np.argmax(far))
    print(f"\n[L{a.L}] NMF R={R} motifs (peak δ, far-mass δ>256):", flush=True)
    for k in range(R):
        tag = "  <-- SPLASH" if k == splash_k else ""
        print(f"  motif {k}: peak_δ={peakd[k]:5d}  far_mass={far[k]:.3f}{tag}", flush=True)

    # ---- GATE-1: can [V,D,P] predict splash-motif usage? ----
    X = np.stack([sink, Dsc, Psc], 1)                # hand-crafted V/D/P vocabulary
    y = U[:, splash_k]
    reg = LinearRegression().fit(X, y)
    r2 = reg.score(X, y)
    # also: best single-length R^2 predicting EACH motif's usage from V/D/P
    r2_all = []
    for k in range(R):
        r2_all.append(LinearRegression().fit(X, U[:, k]).score(X, U[:, k]))
    verdict = ("PASS (flexible basis captures a motif OUTSIDE V/D/P)"
               if (far[splash_k] > 0.15 and peakd[splash_k] > 64 and r2 < 0.5)
               else "WEAK/FAIL (splash motif absent or explainable by V/D/P)")
    print(f"\n[L{a.L}] GATE-1: [V,D,P]->splash-usage R^2 = {r2:.3f}  "
          f"(splash far_mass={far[splash_k]:.3f}, peak_δ={peakd[splash_k]})", flush=True)
    print(f"[L{a.L}] per-motif [V,D,P] R^2: " +
          " ".join(f"{k}:{r2_all[k]:.2f}" for k in range(R)), flush=True)
    print(f"[L{a.L}] VERDICT: {verdict}", flush=True)

    # ---- save CSV + figure ----
    import csv
    with open(os.path.join(a.out, f"offset_motif_L{a.L}.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["layer", "head", "V_sink", "D_local", "P_period"] +
                                      [f"usage_m{k}" for k in range(R)])
        for idx in range(len(G)):
            l, h = idx // nh, idx % nh
            w.writerow([l, h, f"{sink[idx]:.4f}", f"{Dsc[idx]:.4f}", f"{Psc[idx]:.4f}"] +
                       [f"{U[idx,k]:.4f}" for k in range(R)])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    dd = np.arange(T)
    for k in range(R):
        axes[0].plot(dd, M[k], lw=1.6, label=f"m{k}" + (" SPLASH" if k == splash_k else ""))
    axes[0].axvline(512, color="k", ls="--", lw=0.8); axes[0].set_xlabel("offset δ")
    axes[0].set_title(f"L{a.L} offset-motif basis (NMF R={R})"); axes[0].legend(fontsize=8)
    order = np.argsort(-U[:, splash_k])
    im = axes[1].imshow(G[order], aspect="auto", cmap="hot",
                        extent=[0, T, len(G), 0], interpolation="nearest")
    axes[1].axvline(512, color="cyan", ls="--", lw=0.8)
    axes[1].set_xlabel("offset δ"); axes[1].set_ylabel("head (sorted by splash usage)")
    axes[1].set_title("offset profiles g_h(δ)"); fig.colorbar(im, ax=axes[1], fraction=0.046)
    axes[2].scatter(np.maximum(sink, np.maximum(Dsc, Psc)), y, s=10, alpha=0.5)
    axes[2].set_xlabel("max(V,D,P) hand score"); axes[2].set_ylabel("splash-motif usage")
    axes[2].set_title(f"GATE-1: V/D/P vs splash usage (R^2={r2:.2f})")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, f"offset_motif_L{a.L}.png"),
                                    dpi=130, bbox_inches="tight")
    print(f"saved -> {a.out}/offset_motif_L{a.L}.png", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--L", type=int, default=2048)
    ap.add_argument("--n_cases", type=int, default=20)
    ap.add_argument("--R", type=int, default=5)
    ap.add_argument("--out", required=True)
    run(ap.parse_args())

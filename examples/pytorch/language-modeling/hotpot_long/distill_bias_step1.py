#!/usr/bin/env python3
"""distill_bias_step1.py — extract a training-length-distilled 2D structural bias.

Idea (the direction): replace RoPE with NoPE content-QK + a FROZEN additive logit bias
b(i,j) DISTILLED from what the model learned at its training length. Unlike the whole
relative-bias family (ALiBi/T5/KERPLE/FIRE — all functions of i-j only), the distilled
bias contains a NON-relative sink/vertical component (function of absolute key pos j) that
those cannot express.

Step 1 (this script, zero training): from the rotary ckpt at L=512, capture the
content-averaged pre-softmax logit template B_h(i,j) per (layer,head), then fit the
length-extendable additive model
      B_h(i,j) ≈ s_h(δ=i-j)      [slash / offset factor — extends via decaying tail]
              + v_h(j)           [sink / absolute-column factor — extends (BOS at j~0)]
Report how well slash+sink explains the learned structure, and save the frozen tables
s_h[δ], v_h[j] for the Step-2 finetune (NoPE-QK + frozen b).
"""
import argparse, json, os, types
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_POST = {}
VIZ = [(4, 7), (5, 3), (10, 9), (8, 8)]   # slash, slash, sink, sink (PAT-208)


def render_trained(ex):
    lines = [f"{t}: {s}" for t, sents in ex["context"] for s in sents]
    return f"Context:\n{chr(10).join(lines)}\nQuestion: {ex['question']}\nAnswer:"


def install_post_capture(model):
    for _, mod in model.named_modules():
        re = getattr(mod, "rotary_emb", None)
        if re is None:
            continue
        li = int(getattr(mod, "layer_idx", 0))
        orig = re.rotate_queries_or_keys

        def wrapped(self, t, *a, _li=li, _orig=orig, **kw):
            out = _orig(t, *a, **kw)
            _POST.setdefault(_li, []).append(out.detach().float())
            return out
        re.rotate_queries_or_keys = types.MethodType(wrapped, re)


def additive_fit(B, ii, jj, off, cnt_off, cnt_col, T, iters=8):
    """B[T,T] lower-tri; fit B[i,j] ~ s[i-j] + v[j]. Returns s[T], v[T], var_expl."""
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
    install_post_capture(model)
    nl, nh, T = int(cfg.n_layer), int(cfg.n_head), a.L
    D = cfg.n_embd // nh
    scale = 1.0 / np.sqrt(D)

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

    # accumulate content-averaged logit template B_h[i,j]
    Bsum = torch.zeros(nl, nh, T, T)
    for ci, ids in enumerate(cases):
        _POST.clear()
        with torch.no_grad():
            model(torch.tensor([ids], device=dev))
        for l in range(nl):
            q = _POST[l][0][0]; k = _POST[l][1][0]              # [nh,T,D]
            S = torch.einsum("htd,hsd->hts", q, k) * scale       # [nh,T,T]
            Bsum[l] += S.cpu()
        print(f"  case {ci} done", flush=True)
    B = (Bsum / len(cases)).numpy()                              # [nl,nh,T,T]

    tril = np.tril_indices(T)                                    # j<=i
    ii, jj = tril
    off = ii - jj
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
    ve_all = np.array([r[2] for r in rows])
    slfrac = np.array([r[3] for r in rows])
    print(f"\nadditive slash+sink fit over {nl*nh} heads:")
    print(f"  var-explained: mean={ve_all.mean():.3f}  median={np.median(ve_all):.3f}  "
          f"min={ve_all.min():.3f}")
    print(f"  slash share of explained energy: mean={slfrac.mean():.3f}  "
          f"(sink share mean={1-slfrac.mean():.3f})")
    for (l, h) in VIZ:
        r = rows[l * nh + h]
        print(f"  L{l}H{h}: var_expl={r[2]:.3f} slash_frac={r[3]:.2f} sink_frac={r[4]:.2f}")

    np.savez(os.path.join(a.out, f"distilled_bias_L{a.L}.npz"),
             s=S_tab, v=V_tab, n_layer=nl, n_head=nh, L=T,
             var_explained=ve_all, slash_frac=slfrac)
    import csv
    with open(os.path.join(a.out, f"distilled_bias_L{a.L}.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["layer", "head", "var_explained", "slash_frac", "sink_frac"])
        for r in rows:
            w.writerow([r[0], r[1], f"{r[2]:.4f}", f"{r[3]:.4f}", f"{r[4]:.4f}"])

    # figure: for VIZ heads, show B, reconstruction, s(δ), v(j)
    fig, axes = plt.subplots(len(VIZ), 4, figsize=(16, 4 * len(VIZ)))
    for r, (l, h) in enumerate(VIZ):
        s = S_tab[l * nh + h]; v = V_tab[l * nh + h]
        Bhh = B[l, h].copy()
        recon = s[off] + v[jj]; Rec = np.full((T, T), np.nan); Rec[ii, jj] = recon
        m = np.tril(np.ones((T, T), bool))
        Bshow = np.where(m, Bhh, np.nan)
        vmin, vmax = np.nanpercentile(Bshow, 2), np.nanpercentile(Bshow, 98)
        axes[r][0].imshow(Bshow, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
        axes[r][0].set_title(f"L{l}H{h} B(i,j) [avg logit]", fontsize=9)
        axes[r][1].imshow(Rec, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
        axes[r][1].set_title("recon s(δ)+v(j)", fontsize=9)
        axes[r][2].plot(s, lw=1.0); axes[r][2].set_title("slash s(δ)", fontsize=9)
        axes[r][2].set_xlabel("offset δ")
        axes[r][3].plot(v[:64], lw=1.0); axes[r][3].set_title("sink v(j) [first 64]", fontsize=9)
        axes[r][3].set_xlabel("key pos j")
        for c in range(4):
            axes[r][c].tick_params(labelsize=6)
    fig.suptitle(f"Distilled 2D bias @ L{a.L}: slash(offset)+sink(abs-col) factorization", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(os.path.join(a.out, f"distilled_bias_L{a.L}.png"), dpi=120, bbox_inches="tight")
    print(f"saved -> {a.out}/distilled_bias_L{a.L}.png", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--L", type=int, default=512)
    ap.add_argument("--n_cases", type=int, default=20)
    ap.add_argument("--out", required=True)
    run(ap.parse_args())

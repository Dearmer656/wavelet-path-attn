#!/usr/bin/env python3
"""distill_recon_error.py — Step 1.5: rank heads by reconstruction error at extrapolation.

Uses the L512-distilled motif b_h = s_h(δ)+v_h(j) (from distill_bias_step1.py). At the
extrapolation length, capture each head's content-averaged logit B_h(i,j) and measure how
well the frozen (length-extended) L512 motif reconstructs it. Heads with LARGE recon error
= the "broken" heads (the splash-exploded ones) → candidates for the motif+NoPE-qk
substitution. Good heads (low error) keep qRk untouched.

Extension of the L512 motif to length L: s(δ) held at s(511) for δ>511 (decayed slash
tail); v(j) held at its non-sink baseline for j>=512 (sink lives only in early columns).
"""
import argparse, json, os, types
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_POST = {}


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


def run(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out, exist_ok=True)
    d = np.load(a.motif)
    S0, V0 = d["s"], d["v"]                       # [144, L512]
    L0 = int(d["L"]); nl = int(d["n_layer"]); nh = int(d["n_head"])
    T = a.L
    # extend motif tables to length T
    S = np.zeros((nl * nh, T)); V = np.zeros((nl * nh, T))
    for idx in range(nl * nh):
        S[idx, :L0] = S0[idx]; S[idx, L0:] = S0[idx, L0 - 1]              # hold slash tail
        base = np.median(V0[idx, 64:L0]) if L0 > 64 else 0.0
        V[idx, :L0] = V0[idx]; V[idx, L0:] = base                        # sink baseline

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
    D = cfg.n_embd // nh; scale = 1.0 / np.sqrt(D)

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

    Bsum = torch.zeros(nl, nh, T, T)
    for ci, ids in enumerate(cases):
        _POST.clear()
        with torch.no_grad():
            model(torch.tensor([ids], device=dev))
        for l in range(nl):
            q = _POST[l][0][0]; k = _POST[l][1][0]
            Bsum[l] += (torch.einsum("htd,hsd->hts", q, k) * scale).cpu()
        print(f"  case {ci} done", flush=True)
    B = (Bsum / len(cases)).numpy()

    ii, jj = np.tril_indices(T); off = ii - jj
    ood = off >= L0
    err = np.zeros(nl * nh); err_ood = np.zeros(nl * nh); ood_absmax = np.zeros(nl * nh)
    for l in range(nl):
        for h in range(nh):
            idx = l * nh + h
            b = B[l, h][ii, jj]
            recon = S[idx][off] + V[idx][jj]
            res = b - recon
            ss_tot = ((b - b.mean()) ** 2).sum() + 1e-9
            err[idx] = (res ** 2).sum() / ss_tot                          # 1 - var_expl
            err_ood[idx] = np.sqrt((res[ood] ** 2).mean()) if ood.any() else 0.0
            ood_absmax[idx] = np.abs(b[ood]).max() if ood.any() else 0.0

    order = np.argsort(-err_ood)
    print("\nTOP-20 broken heads (by OOD recon RMSE):")
    print(f"{'rank':>4} {'L':>3} {'H':>3} {'err(1-VE)':>10} {'ood_rmse':>9} {'ood_absmax':>10}")
    for r, idx in enumerate(order[:20]):
        print(f"{r:>4} {idx//nh:>3} {idx%nh:>3} {err[idx]:>10.3f} {err_ood[idx]:>9.3f} "
              f"{ood_absmax[idx]:>10.2f}", flush=True)
    layers = np.array([idx // nh for idx in range(nl * nh)])
    lay_err = np.array([err_ood[layers == l].mean() for l in range(nl)])
    print("\nmean OOD recon RMSE by layer:")
    for l in range(nl):
        print(f"  L{l:>2}: {lay_err[l]:.3f}", flush=True)
    print(f"\ncorr(layer, ood_rmse) = {np.corrcoef(layers, err_ood)[0,1]:.3f} "
          f"(positive => deeper layers more broken)", flush=True)

    import csv
    with open(os.path.join(a.out, f"recon_error_L{a.L}.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["layer", "head", "err_1mVE", "ood_rmse", "ood_absmax"])
        for idx in range(nl * nh):
            w.writerow([idx // nh, idx % nh, f"{err[idx]:.4f}", f"{err_ood[idx]:.4f}",
                        f"{ood_absmax[idx]:.4f}"])

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    im = ax[0].imshow(err_ood.reshape(nl, nh), aspect="auto", cmap="hot", origin="lower")
    ax[0].set_xlabel("head"); ax[0].set_ylabel("layer")
    ax[0].set_title(f"OOD recon RMSE (L{a.L} vs L{L0} motif)\nbright = broken head")
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    ax[1].scatter(layers, err_ood, s=14, alpha=0.5)
    ax[1].plot(range(nl), lay_err, "-o", color="tab:red")
    ax[1].set_xlabel("layer"); ax[1].set_ylabel("OOD recon RMSE")
    ax[1].set_title("broken-ness by depth"); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, f"recon_error_L{a.L}.png"),
                                    dpi=130, bbox_inches="tight")
    print(f"saved -> {a.out}/recon_error_L{a.L}.png", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--motif", required=True, help="distilled_bias_L512.npz")
    ap.add_argument("--L", type=int, default=2048)
    ap.add_argument("--n_cases", type=int, default=20)
    ap.add_argument("--out", required=True)
    run(ap.parse_args())

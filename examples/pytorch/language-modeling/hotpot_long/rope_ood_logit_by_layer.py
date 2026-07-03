#!/usr/bin/env python3
"""rope_ood_logit_by_layer.py — PAT-208 follow-up: direct depth profile of the OOD
"splash explosion".

For every (layer, head) capture POST-RoPE q/k at L=2048 (train=512) and measure the
out-of-distribution large-offset pre-softmax logit:
  ood_max(i)   = max_{j : i-j >= train} S[i, j]      (i >= train)
  indist_max(i)= max_{j : 1 <= i-j < train} S[i, j]
per head we summarise over the OOD query rows (i >= train):
  OOD_MAX   = max_i ood_max(i)        (the extreme-value spike that wins the softmax)
  OOD_P99   = 99th pct of ood_max(i)
  INDIST_MAX= max_i indist_max(i)     (in-distribution reference, should be ~flat over depth)
Aggregated over N cases. Output: a per-layer curve (max & mean over heads) of OOD_MAX
vs the in-distribution reference -> shows shallow layers barely explode, late layers
(8-11) explode. Also dumps a per-(layer,head) CSV so the hijack heads are identifiable.
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

        def wrapped(self, t, *args, _li=li, _orig=orig, **kw):
            out = _orig(t, *args, **kw)              # POST-RoPE
            _POST.setdefault(_li, []).append(out.detach().float().cpu())
            return out
        re.rotate_queries_or_keys = types.MethodType(wrapped, re)


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

    nl, nh = int(cfg.n_layer), int(cfg.n_head)
    T, tr = a.L, a.train
    D = model.config.n_embd // nh
    scale = 1.0 / np.sqrt(D)

    cases = select_cases(tok, a.jsonl, a.L, a.n_cases)
    print(f"selected {len(cases)} cases at L{a.L}", flush=True)

    # accumulators [n_cases, nl, nh]
    OOD_MAX = np.full((len(cases), nl, nh), np.nan)
    OOD_P99 = np.full((len(cases), nl, nh), np.nan)
    INDIST  = np.full((len(cases), nl, nh), np.nan)

    for ci, sel in enumerate(cases):
        _POST.clear()
        with torch.no_grad():
            model(torch.tensor([sel], device=dev))
        for l in range(nl):
            q = _POST[l][0][0]; k = _POST[l][1][0]      # [nh,T,D]
            for h in range(nh):
                S = (q[h].numpy() @ k[h].numpy().T) * scale
                ood = np.full(T, np.nan); ind = np.full(T, np.nan)
                for i in range(1, T):
                    lo_in = max(0, i - (tr - 1))
                    ind[i] = S[i, lo_in:i].max()
                    if i >= tr:
                        ood[i] = S[i, :i - (tr - 1)].max()
                OOD_MAX[ci, l, h] = np.nanmax(ood[tr:])
                OOD_P99[ci, l, h] = np.nanpercentile(ood[tr:], 99)
                INDIST[ci, l, h]  = np.nanmax(ind[tr:])
        print(f"case {ci} done", flush=True)

    # mean over cases -> [nl, nh]
    ood_max_c = np.nanmean(OOD_MAX, axis=0)
    ood_p99_c = np.nanmean(OOD_P99, axis=0)
    indist_c  = np.nanmean(INDIST, axis=0)

    # per-layer aggregates
    layer_ood_max_over_heads  = np.nanmax(ood_max_c, axis=1)   # worst head per layer
    layer_ood_mean_over_heads = np.nanmean(ood_max_c, axis=1)
    layer_indist_max          = np.nanmax(indist_c, axis=1)

    # CSV per (layer,head)
    import csv
    with open(os.path.join(a.out, "ood_logit_by_layer_head.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "head", "ood_max", "ood_p99", "indist_max", "ood_minus_indist"])
        for l in range(nl):
            for h in range(nh):
                w.writerow([l, h, f"{ood_max_c[l,h]:.3f}", f"{ood_p99_c[l,h]:.3f}",
                            f"{indist_c[l,h]:.3f}", f"{ood_max_c[l,h]-indist_c[l,h]:.3f}"])

    # ---- figure ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    xs = np.arange(nl)
    ax1.plot(xs, layer_ood_max_over_heads, "-o", color="tab:red", lw=2,
             label="OOD max logit (worst head / layer)")
    ax1.plot(xs, layer_ood_mean_over_heads, "-s", color="tab:orange", lw=1.5,
             label="OOD max logit (mean over heads)")
    ax1.plot(xs, layer_indist_max, "--^", color="tab:blue", lw=1.5,
             label="in-dist max logit (reference, worst head)")
    ax1.axvspan(7.5, 11.5, color="red", alpha=0.08)
    ax1.text(9.5, ax1.get_ylim()[1]*0.05, "layers 8-11\n(NoPE rescue)", ha="center",
             fontsize=8, color="darkred")
    ax1.set_xlabel("layer"); ax1.set_ylabel("pre-softmax logit")
    ax1.set_title(f"OOD-offset logit 'splash explosion' by depth\n(L{a.L}, train={tr}, "
                  f"mean over {len(cases)} cases)", fontsize=10)
    ax1.set_xticks(xs); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    # per-(layer,head) heatmap of OOD-minus-indist (the hijack margin)
    im = ax2.imshow((ood_max_c - indist_c).T, aspect="auto", cmap="hot",
                    origin="lower", interpolation="nearest")
    ax2.set_xlabel("layer"); ax2.set_ylabel("head")
    ax2.set_title("OOD_max − in-dist_max  (per head; >0 = OOD spike out-competes local slash)",
                  fontsize=10)
    ax2.set_xticks(xs)
    fig.colorbar(im, ax=ax2, fraction=0.046)
    fig.tight_layout()
    out_png = os.path.join(a.out, "ood_logit_by_layer.png")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"saved -> {out_png}")

    # text summary
    print("\nlayer | OOD_max(worst) | OOD_max(mean) | indist_max(worst)")
    for l in range(nl):
        print(f"L{l:>2}  | {layer_ood_max_over_heads[l]:>13.2f} | "
              f"{layer_ood_mean_over_heads[l]:>12.2f} | {layer_indist_max[l]:>16.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--L", type=int, default=2048)
    ap.add_argument("--train", type=int, default=512)
    ap.add_argument("--n_cases", type=int, default=5)
    ap.add_argument("--out", required=True)
    run(ap.parse_args())

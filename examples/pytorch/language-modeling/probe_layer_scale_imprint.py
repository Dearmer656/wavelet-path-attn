#!/usr/bin/env python3
"""
Refined hypothesis (user): different LAYERS use different scales, and each layer's
preferred scale shapes THAT layer's attention/backbone. Layer-averaging (the
cross-ρ probe) washes this out — so test it PER LAYER on the K=8 model.

For the K=8 QWAB model, per layer ℓ:
  - router preference: ρ*(ℓ)=argmax_s π_ℓ[s]; π-weighted geo-mean scale ρ̄(ℓ).
  - backbone (E_base_raw) spectrum P_ℓ(f); spectral centroid cen(ℓ).
Prediction if per-layer scale shapes per-layer backbone:
  (1) layers preferring COARSER scales have LOWER-freq backbones:
      corr over layers (log ρ*(ℓ), log cen(ℓ)) < 0.
  (2) each layer has EXCESS backbone power near f_peak(ρ*(ℓ)) vs the cross-layer
      mean spectrum at that freq: δ_ℓ(f_peak(ρ*(ℓ))) > 0.
Control: same test on a K=1 model (all layers share one scale) — there the
per-layer centroid spread should NOT track any (absent) per-layer scale signal.
"""
import json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/project/nlp-work5/hongyu-s/transformers/src")
sys.path.insert(0, "/project/nlp-work5/hongyu-s/flash-linear-attention")
import fla.layers.path_attn as pa
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

K8 = "/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/head_wise_scale_selection_vs_layer_wise/layer_wise/sigmoid_exp/s42_delta_detach/checkpoint-15000"
HOTPOT = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/hotpot_long_dev_uniform.jsonl"
OUT = Path("/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat234_scale_card")
L = 4096
DMAX = 4000
Q_LO = 512
N_SAMPLES = 4
SCALES = np.array([1, 4, 16, 64, 256, 1024, 4096, 16384], float)
F_PEAK = np.sqrt(2) / (2 * np.pi * SCALES)


def render(q, ctx):
    c = "".join(f"Title: {t}\n{' '.join(s).strip()}\n\n" for t, s in ctx)
    return f"Context:\n{c}\nQuestion: {q}\nAnswer:"


def spectrum(sig):
    x = np.asarray(sig[1:], float); x = x[np.isfinite(x)]
    if len(x) < 8: return np.array([]), np.array([])
    x = x - x.mean(); w = np.hanning(len(x))
    P = np.abs(np.fft.rfft(x * w)) ** 2
    f = np.fft.rfftfreq(len(x), d=1.0)
    return f, P


def main():
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(K8); cfg.attn_implementation = "path_attn"
    cfg.wavelet_logit_bias_log_every = 10 ** 9; cfg.wavelet_ctxscale_eval_log_once = False
    model = AutoModelForCausalLM.from_pretrained(K8, config=cfg, torch_dtype=torch.float32).to(dev).eval()
    layers = sorted(int(getattr(m, "layer_idx")) for m in model.modules()
                    if isinstance(m, pa.PaTHAttention) and getattr(m, "layer_idx", None) is not None)

    store = {}
    orig = pa.PaTHAttention._build_ctxscale_shift_logit_bias_v0
    def wrapped(self, **kw):
        out = orig(self, **kw)
        base = kw.get("E_base_raw", None)
        lid = int(getattr(self, "layer_idx", 0))
        rp = getattr(self, "_last_ctxscale_router_prob", None)
        store[lid] = (base.detach().float().cpu() if base is not None else None,
                      rp.detach().float().cpu() if rp is not None else None)
        return out
    pa.PaTHAttention._build_ctxscale_shift_logit_bias_v0 = wrapped

    exs = []
    with open(HOTPOT) as f:
        for line in f:
            ex = json.loads(line)
            if ex["meta"]["target_total_tokens"] == 4096: exs.append(ex)
            if len(exs) >= N_SAMPLES: break

    Dm = min(DMAX, L - 1)
    abar = {l: np.zeros(Dm + 1) for l in layers}; abcnt = {l: np.zeros(Dm + 1) for l in layers}
    pi = {l: np.zeros(len(SCALES)) for l in layers}; pin = {l: 0 for l in layers}
    for ex in exs:
        store.clear()
        ids = tok(render(ex["question"], ex["context"]), return_tensors="pt", truncation=False)["input_ids"][:, :L].to(dev)
        with torch.no_grad(): model(input_ids=ids)
        for lid, (base, rp) in store.items():
            if base is not None:
                e = base[0]
                if e.dim() == 2: e = e.unsqueeze(0)
                H, T, _ = e.shape; en = e.numpy()
                for i in range(Q_LO, T):
                    dm = min(Dm, i)
                    abar[lid][1:dm + 1] += en[:, i, i - np.arange(1, dm + 1)].mean(0)
                    abcnt[lid][1:dm + 1] += 1
            if rp is not None:
                r = rp[0]
                while r.dim() > 1: r = r.mean(0)
                pi[lid] += r.numpy()[:len(SCALES)]; pin[lid] += 1
    pa.PaTHAttention._build_ctxscale_shift_logit_bias_v0 = orig

    # per-layer spectrum (normalized AC), centroid, preferred scale
    P, cen, fgrid = {}, {}, None
    for lid in layers:
        f, Pw = spectrum(abar[lid] / np.maximum(abcnt[lid], 1))
        if f.size == 0: continue
        fgrid = f; Pn = Pw / max(Pw.sum(), 1e-12); P[lid] = Pn
        fp = f > 0
        cen[lid] = float((f[fp] * Pn[fp]).sum() / max(Pn[fp].sum(), 1e-12))
    Pbar = np.mean([P[l] for l in P], axis=0)

    print(f"=== per-layer: preferred scale vs backbone spectrum (K=8, L{L}, {len(exs)} samples) ===")
    print(f"{'L':>3} | {'rho*':>6} {'rho_bar':>8} | {'centroid':>9} | {'f_peak(rho*)':>12} | {'delta@f_peak':>12}")
    rows = []
    for lid in layers:
        if lid not in P: continue
        p = pi[lid] / max(pin[lid], 1); p = p / max(p.sum(), 1e-12)
        rstar = SCALES[int(np.argmax(p))]
        rbar = float(np.exp((p * np.log(SCALES)).sum()))          # pi-weighted geo-mean scale
        fp = np.sqrt(2) / (2 * np.pi * rstar)
        dloc = float(np.interp(fp, fgrid, P[lid] - Pbar))
        rows.append((lid, rstar, rbar, cen[lid], fp, dloc))
        print(f"{lid:>3} | {int(rstar):>6} {rbar:>8.1f} | {cen[lid]:>9.3e} | {fp:>12.2e} | {dloc:>+12.4f}")

    rstar = np.array([r[1] for r in rows]); rbar = np.array([r[2] for r in rows])
    cc = np.array([r[3] for r in rows]); dloc = np.array([r[5] for r in rows])
    c1 = float(np.corrcoef(np.log(rstar), np.log(cc))[0, 1])
    c2 = float(np.corrcoef(np.log(rbar), np.log(cc))[0, 1])
    print(f"\n(1) corr over layers (log rho*,  log centroid) = {c1:+.3f}   (<0 => coarse-pref layers have lower-freq backbone)")
    print(f"    corr over layers (log rho_bar, log centroid) = {c2:+.3f}")
    print(f"(2) delta@f_peak(rho*) > 0 in {(dloc > 0).sum()}/{len(dloc)} layers  (mean {dloc.mean():+.4f})")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        fig, ax = plt.subplots(1, 2, figsize=(17, 6.5))
        fp = fgrid > 0
        norm = plt.Normalize(np.log2(rstar).min(), np.log2(rstar).max())
        for r in rows:
            lid, rs = r[0], r[1]
            col = cm.plasma(norm(np.log2(rs)))
            ax[0].loglog(fgrid[fp], P[lid][fp], lw=0.9, color=col, alpha=0.85)
            ax[0].axvline(r[4], color=col, ls=":", lw=0.7)
        sm = cm.ScalarMappable(cmap="plasma", norm=norm); sm.set_array([])
        cb = fig.colorbar(sm, ax=ax[0]); cb.set_label("log2(preferred scale ρ*)")
        ax[0].set_title("per-layer backbone spectrum, colored by that layer's preferred scale\n(dotted = f_peak(ρ*) of same color)")
        ax[0].set_xlabel("f (cyc/sample)"); ax[0].set_ylabel("norm AC power")
        ax[1].scatter(rstar, cc, c=np.log2(rstar), cmap="plasma", s=90)
        for r in rows: ax[1].annotate(f"L{r[0]}", (r[1], r[3]), fontsize=7)
        ax[1].set_xscale("log"); ax[1].set_yscale("log")
        ax[1].set_xlabel("preferred scale ρ*(ℓ)"); ax[1].set_ylabel("backbone spectral centroid")
        ax[1].set_title(f"per-layer: preferred scale vs backbone centroid\ncorr(logρ*,log cen)={c1:+.3f}")
        out = OUT / "layer_scale_imprint.png"
        fig.tight_layout(); fig.savefig(out, dpi=140)
        print(f"\nsaved: {out}")
    except Exception as e:
        print(f"(plot failed: {e})")


if __name__ == "__main__":
    main()

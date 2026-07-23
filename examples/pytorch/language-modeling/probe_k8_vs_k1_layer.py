#!/usr/bin/env python3
"""
CAUSAL control for "per-layer scale selection shapes per-layer backbone".

Compare, per layer, the PaTH backbone spectrum of:
  K8 = s42_delta_detach (router gives layer ℓ its own preferred scale ρ*(ℓ))
  K1 = S1_s42           (ALL layers forced to the single scale ρ128)
Same seed42 / init / data / steps; the ONLY difference is whether each layer may
choose its own scale. So δ_ℓ(f) = P^{K8}_ℓ(f) − P^{K1}_ℓ(f) isolates what the
per-layer scale DIFFERENTIATION did to layer ℓ's backbone, on top of the common
ρ128 baseline.

Prediction if layer-specific scale shapes the layer:
  layers whose K8 preference ρ*(ℓ) is COARSER than ρ128 (e.g. L1→ρ16384, L2→ρ4096)
  should gain LOW-freq backbone power in K8 vs K1 => δ_ℓ > 0 near f_peak(ρ*(ℓ));
  layers with ρ*(ℓ) ≈ ρ128 => δ_ℓ ≈ 0.
Summary: corr over layers between log(ρ*_K8/ρ128) and (K8−K1 low-freq power shift).
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

RUNS = Path("/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs")
K8 = RUNS / "head_wise_scale_selection_vs_layer_wise/layer_wise/sigmoid_exp/s42_delta_detach/checkpoint-15000"
K1 = RUNS / "pat225_scale_card/S1_s42/checkpoint-15000"     # single scale rho128
HOTPOT = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/hotpot_long_dev_uniform.jsonl"
OUT = Path("/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat234_scale_card")
L = 4096; DMAX = 4000; Q_LO = 512; N_SAMPLES = 4
RHO_K1 = 128.0
SCALES = np.array([1, 4, 16, 64, 256, 1024, 4096, 16384], float)
F_LOW = 1.0 / L * np.sqrt(2)   # "low-freq band" edge ~ around f_train scale; use <= a few/L


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


def run_model(ckpt, tok, exs, dev, want_router):
    cfg = AutoConfig.from_pretrained(ckpt); cfg.attn_implementation = "path_attn"
    cfg.wavelet_logit_bias_log_every = 10 ** 9; cfg.wavelet_ctxscale_eval_log_once = False
    model = AutoModelForCausalLM.from_pretrained(ckpt, config=cfg, torch_dtype=torch.float32).to(dev).eval()
    layers = sorted(int(getattr(m, "layer_idx")) for m in model.modules()
                    if isinstance(m, pa.PaTHAttention) and getattr(m, "layer_idx", None) is not None)
    store = {}
    orig = pa.PaTHAttention._build_ctxscale_shift_logit_bias_v0
    def wrapped(self, **kw):
        out = orig(self, **kw)
        base = kw.get("E_base_raw", None); lid = int(getattr(self, "layer_idx", 0))
        rp = getattr(self, "_last_ctxscale_router_prob", None)
        store[lid] = (base.detach().float().cpu() if base is not None else None,
                      rp.detach().float().cpu() if (want_router and rp is not None) else None)
        return out
    pa.PaTHAttention._build_ctxscale_shift_logit_bias_v0 = wrapped
    Dm = min(DMAX, L - 1)
    ab = {l: np.zeros(Dm + 1) for l in layers}; cn = {l: np.zeros(Dm + 1) for l in layers}
    pi = {l: np.zeros(len(SCALES)) for l in layers}; pn = {l: 0 for l in layers}
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
                    ab[lid][1:dm + 1] += en[:, i, i - np.arange(1, dm + 1)].mean(0); cn[lid][1:dm + 1] += 1
            if want_router and rp is not None:
                r = rp[0]
                while r.dim() > 1: r = r.mean(0)
                pi[lid] += r.numpy()[:len(SCALES)]; pn[lid] += 1
    pa.PaTHAttention._build_ctxscale_shift_logit_bias_v0 = orig
    del model; torch.cuda.empty_cache()
    P, fgrid = {}, None
    for lid in layers:
        f, Pw = spectrum(ab[lid] / np.maximum(cn[lid], 1))
        if f.size == 0: continue
        fgrid = f; P[lid] = Pw / max(Pw.sum(), 1e-12)
    prefs = {}
    if want_router:
        for lid in layers:
            if pn[lid] == 0: continue
            p = pi[lid] / max(pn[lid], 1); p = p / max(p.sum(), 1e-12)
            prefs[lid] = SCALES[int(np.argmax(p))]
    return P, fgrid, layers, prefs


def main():
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    exs = []
    with open(HOTPOT) as f:
        for line in f:
            ex = json.loads(line)
            if ex["meta"]["target_total_tokens"] == 4096: exs.append(ex)
            if len(exs) >= N_SAMPLES: break

    print("K8 (router, per-layer scales) ...", flush=True)
    P8, fg, layers, prefs = run_model(K8, tok, exs, dev, want_router=True)
    print("K1 (single rho128) ...", flush=True)
    P1, fg1, _, _ = run_model(K1, tok, exs, dev, want_router=False)

    f_peak = lambda r: np.sqrt(2) / (2 * np.pi * r)
    low = fg <= 5.0 / L                        # low-freq band (coarse-only)
    print(f"\n=== K8 vs K1 per-layer backbone shift (low-freq band f<=5/L={5.0/L:.1e}) ===")
    print(f"{'L':>3} | {'rho*_K8':>8} {'rho*/128':>9} | {'δ_lowband(K8-K1)':>16} | {'δ@f_peak(rho*)':>15}")
    rows = []
    for lid in layers:
        if lid not in P8 or lid not in P1: continue
        d = P8[lid] - P1[lid]
        dlow = float(d[low].sum())
        rs = prefs.get(lid, np.nan)
        dfp = float(np.interp(f_peak(rs), fg, d)) if np.isfinite(rs) else np.nan
        rows.append((lid, rs, dlow, dfp))
        print(f"{lid:>3} | {int(rs):>8} {rs/RHO_K1:>9.2f} | {dlow:>+16.4f} | {dfp:>+15.4f}")

    rs = np.array([r[1] for r in rows]); dlow = np.array([r[2] for r in rows])
    lr = np.log(rs / RHO_K1)
    c = float(np.corrcoef(lr, dlow)[0, 1])
    coarse = lr > 0; fine = lr < 0
    print(f"\ncorr over layers ( log(ρ*_K8/128) , low-band shift K8-K1 ) = {c:+.3f}")
    print(f"  coarse-pref layers (ρ*>128): mean low-band shift = {dlow[coarse].mean():+.4f}  (n={coarse.sum()})")
    print(f"  fine-pref   layers (ρ*<128): mean low-band shift = {dlow[fine].mean():+.4f}  (n={fine.sum()})")
    print("  >> POSITIVE corr & coarse>fine => letting a layer pick a coarser scale ADDS low-freq backbone power (causal imprint)")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        ax[0].scatter(rs / RHO_K1, dlow, c=np.log2(rs), cmap="plasma", s=90)
        for r in rows: ax[0].annotate(f"L{r[0]}", (r[1] / RHO_K1, r[2]), fontsize=8)
        ax[0].axhline(0, color="gray", lw=0.6); ax[0].axvline(1, color="gray", ls="--", lw=0.6)
        ax[0].set_xscale("log"); ax[0].set_xlabel("K8 preferred scale / ρ128 (K1)")
        ax[0].set_ylabel("low-freq backbone power shift  K8 − K1")
        ax[0].set_title(f"does a coarser per-layer choice add low-freq backbone power?\ncorr={c:+.3f}")
        # per-layer difference curves for the coarse-preferring layers
        for r in rows:
            lid, rr = r[0], r[1]
            if rr > 512:   # coarse-preferring layers
                d = P8[lid] - P1[lid]
                ax[1].semilogx(fg[fg > 0], d[fg > 0], lw=1.0, label=f"L{lid} ρ*{int(rr)}")
                ax[1].axvline(f_peak(rr), color="gray", ls=":", lw=0.6)
        ax[1].axhline(0, color="gray", lw=0.6)
        ax[1].set_title("δ_ℓ(f)=P^K8−P^K1 for coarse-preferring layers"); ax[1].legend(fontsize=8)
        ax[1].set_xlabel("f (cyc/sample)"); ax[1].set_ylabel("δ power")
        out = OUT / "k8_vs_k1_layer.png"
        fig.tight_layout(); fig.savefig(out, dpi=140)
        print(f"\nsaved: {out}")
    except Exception as e:
        print(f"(plot failed: {e})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Does QWAB training SHAPE the PaTH backbone in a scale-specific way?

Controlled cross-ρ design: all K=1 no_pe models share the identical recipe
(seed 42, same init/data-order/steps) and differ ONLY in the single trained
wavelet scale ρ. E_base_raw is the PURE PaTH backbone logit (pre-wavelet), so
its spectrum reflects only what TRAINING left in the Q/K/Householder weights.

If training with scale ρ imprints a signature at that scale's Ricker freq center
f_peak(ρ)=√2/(2πρ), then each model's backbone spectrum should show EXCESS power
near its own f_peak(ρ) relative to the cross-model ensemble mean — and the
per-model spectral centroid should shift DOWN monotonically with ρ.

Anchor = ensemble mean over models (no clean no_pe no-wavelet baseline exists;
PA_baseline uses vanilla PE => E_base_raw includes RoPE, not comparable).
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

RUNS = Path("/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat225_scale_card")
HOTPOT = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/hotpot_long_dev_uniform.jsonl"
OUT = Path("/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat234_scale_card")
L = 4096
DMAX = 4000
Q_LO = 512
N_SAMPLES = 4

# (label, rho, ckpt dir) — only ckpt-15000-ready s42 models; extend as more finish.
MODELS = [
    ("me0",    1,     RUNS / "S1_me0_s42"),
    ("me8",    16,    RUNS / "S1_me8_s42"),
    ("S1(me14)", 128, RUNS / "S1_s42"),
    ("me20",   1024,  RUNS / "S1_me20_s42"),
    ("me28",   16384, RUNS / "S1_me28_s42"),
]
CKPT = "checkpoint-15000"


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


def backbone_spectra(ckpt, tok, exs, dev):
    """Return {layer: normalized-AC-power P(f)} and shared freq grid, for one model."""
    cfg = AutoConfig.from_pretrained(ckpt); cfg.attn_implementation = "path_attn"
    cfg.wavelet_logit_bias_log_every = 10 ** 9; cfg.wavelet_ctxscale_eval_log_once = False
    model = AutoModelForCausalLM.from_pretrained(ckpt, config=cfg, torch_dtype=torch.float32).to(dev).eval()
    layers = sorted(int(getattr(m, "layer_idx")) for m in model.modules()
                    if isinstance(m, pa.PaTHAttention) and getattr(m, "layer_idx", None) is not None)
    store = {}
    orig = pa.PaTHAttention._build_ctxscale_shift_logit_bias_v0
    def wrapped(self, **kw):
        out = orig(self, **kw)
        base = kw.get("E_base_raw", None)
        if base is not None:
            store[int(getattr(self, "layer_idx", 0))] = base.detach().float().cpu()
        return out
    pa.PaTHAttention._build_ctxscale_shift_logit_bias_v0 = wrapped
    Dm = min(DMAX, L - 1)
    acc = {l: np.zeros(Dm + 1) for l in layers}; cnt = {l: np.zeros(Dm + 1) for l in layers}
    for ex in exs:
        store.clear()
        ids = tok(render(ex["question"], ex["context"]), return_tensors="pt", truncation=False)["input_ids"][:, :L].to(dev)
        with torch.no_grad(): model(input_ids=ids)
        for lid, e in store.items():
            e = e[0]
            if e.dim() == 2: e = e.unsqueeze(0)
            H, T, _ = e.shape; en = e.numpy()
            for i in range(Q_LO, T):
                dm = min(Dm, i)
                acc[lid][1:dm + 1] += en[:, i, i - np.arange(1, dm + 1)].mean(0)
                cnt[lid][1:dm + 1] += 1
    pa.PaTHAttention._build_ctxscale_shift_logit_bias_v0 = orig
    del model; torch.cuda.empty_cache()
    out, fgrid = {}, None
    for lid in layers:
        abar = acc[lid] / np.maximum(cnt[lid], 1)
        f, P = spectrum(abar)
        if f.size == 0: continue
        fgrid = f
        out[lid] = P / max(P.sum(), 1e-12)          # normalized AC-power distribution
    return out, fgrid, layers


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

    specs, fgrid, layers = {}, None, None
    for label, rho, d in MODELS:
        ck = d / CKPT
        if not ck.exists():
            print(f"[skip] {label} rho{rho}: no {CKPT}"); continue
        print(f"computing backbone spectrum: {label} (rho{rho}) ...", flush=True)
        sp, fg, lys = backbone_spectra(ck, tok, exs, dev)
        specs[(label, rho)] = sp; fgrid = fg; layers = lys
    if len(specs) < 2:
        print("need >=2 models"); return

    f_peak = lambda r: np.sqrt(2) / (2 * np.pi * r)
    keys = list(specs.keys())
    # per-layer ensemble mean, then per-model deviation, aggregated over layers
    dagg = {k: np.zeros_like(fgrid) for k in keys}
    ncnt = 0
    for lid in layers:
        stack = [specs[k][lid] for k in keys if lid in specs[k]]
        if len(stack) < len(keys): continue
        Pbar = np.mean(stack, axis=0)
        for k in keys:
            dagg[k] += specs[k][lid] - Pbar
        ncnt += 1
    for k in keys: dagg[k] /= max(ncnt, 1)

    # per-model aggregate spectrum (mean over layers) + spectral centroid
    Pagg = {k: np.mean([specs[k][l] for l in layers if l in specs[k]], axis=0) for k in keys}
    fpos = fgrid > 0
    print(f"\n=== scale-specific imprint (ensemble-mean anchor, {ncnt} layers, L{L}) ===")
    print("excess = mean(δ) in ±½-octave band around f_peak(ρ), relative to band mean power")
    print(f"{'model':<10} {'rho':>6} {'f_peak':>9} | {'excess@f_peak':>13} | {'centroid':>9}")
    rows = []
    for (label, rho) in keys:
        fp = f_peak(rho)
        band = (fgrid >= fp / np.sqrt(2)) & (fgrid <= fp * np.sqrt(2))
        base_band = Pagg[(label, rho)][band].mean() if band.any() else np.nan
        excess = dagg[(label, rho)][band].mean() / max(base_band, 1e-12) if band.any() else np.nan
        cen = float((fgrid[fpos] * Pagg[(label, rho)][fpos]).sum() / max(Pagg[(label, rho)][fpos].sum(), 1e-12))
        rows.append((rho, cen, excess))
        print(f"{label:<10} {rho:>6} {fp:>9.2e} | {excess:>+13.3f} | {cen:>9.4e}")
    rr = np.array([r[0] for r in rows]); cc = np.array([r[1] for r in rows])
    sp_corr = float(np.corrcoef(np.log(rr), np.log(cc))[0, 1])
    print(f"\ncentroid vs rho: corr(log ρ, log centroid) = {sp_corr:+.3f}   (expect < 0: larger ρ => lower-freq backbone)")
    ex = np.array([r[2] for r in rows])
    print(f"excess@f_peak: {(ex > 0).sum()}/{len(ex)} models have POSITIVE excess at their own scale's freq")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        cols = cm.viridis(np.linspace(0, 0.92, len(keys)))
        fig, ax = plt.subplots(2, 1, figsize=(13, 11))
        for (k, c) in zip(keys, cols):
            label, rho = k
            ax[0].loglog(fgrid[fpos], Pagg[k][fpos], lw=1.0, color=c, label=f"{label} ρ{rho}")
            ax[0].axvline(f_peak(rho), color=c, ls=":", lw=0.8)
        ax[0].set_title(f"PaTH backbone spectrum (mean over layers), per trained scale ρ — dotted = that model's f_peak(ρ)")
        ax[0].set_xlabel("f (cyc/sample)"); ax[0].set_ylabel("norm AC power"); ax[0].legend(fontsize=8)
        for (k, c) in zip(keys, cols):
            label, rho = k
            ax[1].semilogx(fgrid[fpos], dagg[k][fpos], lw=1.0, color=c, label=f"{label} ρ{rho}")
            fp = f_peak(rho); ax[1].axvline(fp, color=c, ls=":", lw=1.1)
            yv = np.interp(fp, fgrid, dagg[k]); ax[1].scatter([fp], [yv], color=c, s=45, zorder=5)
        ax[1].axhline(0, color="gray", lw=0.6)
        ax[1].set_title("deviation from ensemble mean  δ_ρ(f) = P_ρ − mean  (dot = value at that model's f_peak(ρ))")
        ax[1].set_xlabel("f (cyc/sample)"); ax[1].set_ylabel("δ power"); ax[1].legend(fontsize=8)
        out = OUT / "backbone_imprint_vs_scale.png"
        fig.tight_layout(); fig.savefig(out, dpi=140)
        print(f"\nsaved: {out}")
    except Exception as e:
        print(f"(plot failed: {e})")


if __name__ == "__main__":
    main()

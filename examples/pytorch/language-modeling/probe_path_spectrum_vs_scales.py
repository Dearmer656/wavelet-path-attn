#!/usr/bin/env python3
"""
Does the router pick scales that FILL GAPS in the PaTH backbone spectrum
(complementary) or sit on PaTH's own peaks (redundant)?

Per layer: compute the PaTH base positional spectrum P(f) = |FFT ā(Δ)|² (from
E_base_raw), and the router usage π_scale. Each Ricker scale ρ has freq center
f_peak(ρ)=√2/(2πρ). Sample P at each f_peak and correlate with π across scales:
  corr(π_s, P@f_peak_s) < 0  => router prefers scales where PaTH is WEAK (gap-fill)
  corr > 0                    => router reinforces PaTH's own peaks (redundant)

Model = K=8 QWAB (s42_delta_detach). Length L4096 (freq resolution for the mid scales).
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

CKPT = "/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/head_wise_scale_selection_vs_layer_wise/layer_wise/sigmoid_exp/s42_delta_detach/checkpoint-15000"
HOTPOT = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/hotpot_long_dev_uniform.jsonl"
OUT = Path("/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat234_scale_card")
L = 4096
DMAX = 4000
Q_LO = 512
N_SAMPLES = 4
SCALES = [1, 4, 16, 64, 256, 1024, 4096, 16384]
F_PEAK = {s: np.sqrt(2) / (2 * np.pi * s) for s in SCALES}


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
    cfg = AutoConfig.from_pretrained(CKPT); cfg.attn_implementation = "path_attn"
    cfg.wavelet_logit_bias_log_every = 10 ** 9; cfg.wavelet_ctxscale_eval_log_once = False
    model = AutoModelForCausalLM.from_pretrained(CKPT, config=cfg, torch_dtype=torch.float32).to(dev).eval()
    layers = {}
    for m in model.modules():
        if isinstance(m, pa.PaTHAttention):
            _lid = getattr(m, "layer_idx", None)
            if _lid is None: continue
            layers[int(_lid)] = m
    Ln = max(layers) + 1

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
    abar_acc = {l: np.zeros(Dm + 1) for l in layers}; abar_cnt = {l: np.zeros(Dm + 1) for l in layers}
    pi_acc = {l: np.zeros(len(SCALES)) for l in layers}; pi_n = {l: 0 for l in layers}
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
                    abar_acc[lid][1:dm + 1] += en[:, i, i - np.arange(1, dm + 1)].mean(0)
                    abar_cnt[lid][1:dm + 1] += 1
            if rp is not None:
                r = rp[0]
                while r.dim() > 1: r = r.mean(0)
                pi_acc[lid] += r.numpy()[:len(SCALES)]; pi_n[lid] += 1
    pa.PaTHAttention._build_ctxscale_shift_logit_bias_v0 = orig

    print("=== per-layer: router π vs PaTH spectral energy at each scale's f_peak ===")
    print("corr<0 => router fills PaTH's weak bands (complementary); corr>0 => reinforces PaTH peaks")
    rows_corr = []
    percell = {}
    for lid in sorted(layers):
        abar = abar_acc[lid] / np.maximum(abar_cnt[lid], 1)
        f, P = spectrum(abar)
        if f.size == 0: continue
        pi = pi_acc[lid] / max(pi_n[lid], 1)
        Pn = P / max(P.max(), 1e-12)
        pe = np.array([np.interp(F_PEAK[s], f, Pn) for s in SCALES])   # PaTH energy at each f_peak
        percell[lid] = (pi, pe)
        # correlation across the 8 scales
        if pi.std() > 1e-9 and pe.std() > 1e-9:
            c = float(np.corrcoef(pi, pe)[0, 1]); rows_corr.append(c)
        else:
            c = float("nan")
        top = SCALES[int(np.argmax(pi))]
        print(f"  L{lid:>2} | corr(π, PaTH@f_peak)={c:+.3f} | top-π scale=ρ{top}")
    if rows_corr:
        print(f"  AGG mean corr = {np.nanmean(rows_corr):+.3f}")

    # per-scale aggregate: mean π and mean PaTH energy at f_peak, across layers
    print("\n=== per-scale (mean over layers): f_peak | mean π | mean PaTH-energy@f_peak ===")
    piM = np.mean([percell[l][0] for l in percell], axis=0)
    peM = np.mean([percell[l][1] for l in percell], axis=0)
    for j, s in enumerate(SCALES):
        print(f"  ρ{s:<6d} | f_peak={F_PEAK[s]:.2e} | π={piM[j]:.3f} | PaTH@f_peak={peM[j]:.3f}")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 4, figsize=(18, 11))
        for ax, lid in zip(axes.flat, sorted(layers)):
            abar = abar_acc[lid] / np.maximum(abar_cnt[lid], 1)
            f, P = spectrum(abar); Pn = P / max(P.max(), 1e-12)
            pi = pi_acc[lid] / max(pi_n[lid], 1)
            ax.loglog(f[1:], Pn[1:], lw=0.7, color="steelblue")
            for j, s in enumerate(SCALES):
                fp = F_PEAK[s]; yp = np.interp(fp, f, Pn)
                ax.scatter([fp], [max(yp, 1e-6)], s=20 + 900 * pi[j], c="crimson", alpha=0.6, zorder=3)
                ax.text(fp, max(yp, 1e-6), f"ρ{s}", fontsize=5, ha="center")
            ax.set_title(f"L{lid}  (marker∝π)", fontsize=8); ax.set_xlabel("f", fontsize=7)
        fig.suptitle("PaTH backbone spectrum |FFT ā(Δ)|² per layer, with Ricker f_peak markers sized by router π (K=8, L4096)")
        out = OUT / "path_spectrum_vs_scales.png"
        fig.tight_layout(); fig.savefig(out, dpi=130)
        print(f"\nsaved: {out}")
    except Exception as e:
        print(f"(plot failed: {e})")


if __name__ == "__main__":
    main()

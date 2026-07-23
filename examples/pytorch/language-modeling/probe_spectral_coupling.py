#!/usr/bin/env python3
"""
Spectral-coupling verification for the frequency-domain proof that QWAB's wavelet
subspace cannot cover PaTH's length-extrapolation drift.

Claims to test empirically:
  (i)   PaTH's base attention logit as a function of relative distance, abar(Δ),
        has oscillatory / structured frequency content (not flat).
  (ii)  The length drift δ(Δ) = abar_long(Δ) − abar_short(Δ) concentrates at LOW
        frequency (f < f_train = 1/L_train), i.e. the under-sampled band.
  (iii) That low-frequency band is covered only by COARSE Ricker scales
        (f_peak(ρ)=√2/(2πρ) < f_train ⇒ ρ > √2 L/(2π)), which are the
        softmax-invisible / harmful ones.

Method: capture E_base_raw (PaTH base logits) at L=512 and L=4096 from the QWAB
model, average over (query, head, sample) to abar(Δ), FFT (mean-removed = AC, the
softmax-relevant part), and overlay the 8 Ricker peak-frequencies + f_train.
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
L_SHORT, L_LONG = 512, 4096
DMAX = 4000          # max relative distance to profile at L_LONG
N_SAMPLES = 4
Q_LO = 256           # only use query positions >= Q_LO (avoid warmup) for averaging
SCALES = [1, 4, 16, 64, 256, 1024, 4096, 16384]


def render(q, ctx):
    c = "".join(f"Title: {t}\n{' '.join(s).strip()}\n\n" for t, s in ctx)
    return f"Context:\n{c}\nQuestion: {q}\nAnswer:"


def abar_at_length(model, tok, exs, dev, L, layers):
    """Return {lid: abar[Δ]} for Δ=1..Dmax(L). abar = mean over (query>=Q_LO, head, sample)."""
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
    acc = {l: np.zeros(Dm + 1) for l in layers}
    cnt = {l: np.zeros(Dm + 1) for l in layers}
    for ex in exs:
        store.clear()
        ids = tok(render(ex["question"], ex["context"]), return_tensors="pt", truncation=False)["input_ids"][:, :L].to(dev)
        if ids.shape[1] < Q_LO + 8:
            continue
        with torch.no_grad():
            model(input_ids=ids)
        for lid, e in store.items():
            e = e[0]                                # [H,T,T] or [T,T]
            if e.dim() == 2: e = e.unsqueeze(0)
            H, T, _ = e.shape
            en = e.numpy()
            qlo = max(Q_LO, 1)
            for i in range(qlo, T):
                dmax_i = min(Dm, i)
                # logit from query i to key i-Δ, Δ=1..dmax_i, mean over heads
                vals = en[:, i, i - np.arange(1, dmax_i + 1)].mean(axis=0)  # [dmax_i]
                acc[lid][1:dmax_i + 1] += vals
                cnt[lid][1:dmax_i + 1] += 1
    pa.PaTHAttention._build_ctxscale_shift_logit_bias_v0 = orig
    return {l: acc[l] / np.maximum(cnt[l], 1) for l in layers}


def spectrum(sig):
    """AC power spectrum (mean removed) of sig[Δ], Δ=1..N. Returns freqs (cyc/sample), power."""
    x = np.asarray(sig[1:], dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 8:
        return np.array([]), np.array([])
    x = x - x.mean()
    win = np.hanning(n)
    X = np.fft.rfft(x * win)
    f = np.fft.rfftfreq(n, d=1.0)
    P = (np.abs(X) ** 2)
    return f, P


def main():
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(CKPT); cfg.attn_implementation = "path_attn"
    cfg.wavelet_logit_bias_log_every = 10 ** 9; cfg.wavelet_ctxscale_eval_log_once = False
    model = AutoModelForCausalLM.from_pretrained(CKPT, config=cfg, torch_dtype=torch.float32).to(dev).eval()
    layers = sorted(int(getattr(m, "layer_idx", 0)) for m in model.modules() if isinstance(m, pa.PaTHAttention))
    L = max(layers) + 1

    exs = []
    with open(HOTPOT) as f:
        for line in f:
            ex = json.loads(line)
            if ex["meta"]["target_total_tokens"] == 4096:
                exs.append(ex)
            if len(exs) >= N_SAMPLES: break

    print("computing abar(Δ) at L=512 ...", flush=True)
    ab_s = abar_at_length(model, tok, exs, dev, L_SHORT, layers)
    print("computing abar(Δ) at L=4096 ...", flush=True)
    ab_l = abar_at_length(model, tok, exs, dev, L_LONG, layers)

    f_train = 1.0 / L_SHORT
    f_ricker = {s: np.sqrt(2) / (2 * np.pi * s) for s in SCALES}     # peak freq (cyc/sample)
    rho_star = np.sqrt(2) * L_SHORT / (2 * np.pi)
    print(f"\n f_train (1/L) = {f_train:.3e} cyc/sample ; rho* (f_peak=f_train) = {rho_star:.1f}")
    print(" Ricker peak freqs:  " + "  ".join(f"ρ{s}={f_ricker[s]:.2e}" for s in SCALES))
    print(f" => scales with f_peak < f_train (cover the drift's low-freq band): " +
          ", ".join(f"ρ{s}" for s in SCALES if f_ricker[s] < f_train))

    # aggregate drift spectrum over layers; report low-frequency fraction of drift energy
    print("\n=== per-layer: fraction of DRIFT δ(Δ) energy below f_train (low-freq / coarse-only band) ===")
    lowfrac = []
    drift_specs = {}
    for lid in layers:
        Dov = min(len(ab_s[lid]), len(ab_l[lid])) - 1
        delta = ab_l[lid][1:Dov + 1] - ab_s[lid][1:Dov + 1]
        fδ, Pδ = spectrum(np.concatenate([[0], delta]))
        if fδ.size == 0:
            print(f"  L{lid:>2} | (too short)"); continue
        drift_specs[lid] = (fδ, Pδ)
        lo = Pδ[fδ < f_train].sum() / max(Pδ.sum(), 1e-12)
        lowfrac.append(lo)
        print(f"  L{lid:>2} | low-freq(<f_train) energy fraction = {lo:.3f}")
    if lowfrac:
        print(f"  AGGREGATE mean low-freq drift fraction = {np.mean(lowfrac):.3f}")
        print("  >> high fraction => drift lives in the sub-training-length band => coverable only by coarse ρ")

    # plots
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        # (a) abar(Δ) profile, a representative mid layer
        lmid = layers[len(layers) // 2]
        axes[0, 0].plot(ab_l[lmid][1:600], lw=0.8)
        axes[0, 0].set_title(f"PaTH base logit vs distance  ā(Δ)  (L{lmid}, L=4096)")
        axes[0, 0].set_xlabel("relative distance Δ"); axes[0, 0].set_ylabel("ā(Δ)")
        # (b) spectrum of ā(Δ) at L4096 with Ricker bands
        fL, PL = spectrum(ab_l[lmid])
        axes[0, 1].loglog(fL[1:], PL[1:], lw=0.8, label=f"|FFT ā(Δ)|² (L{lmid})")
        for s in SCALES:
            axes[0, 1].axvline(f_ricker[s], color="gray", ls=":", lw=0.6)
            axes[0, 1].text(f_ricker[s], PL[1:].max()*0.5, f"ρ{s}", rotation=90, fontsize=6, va="top")
        axes[0, 1].axvline(f_train, color="red", ls="--", lw=1, label="f_train=1/512")
        axes[0, 1].set_title("PaTH positional spectrum + Ricker peak-freqs"); axes[0, 1].legend(fontsize=7)
        axes[0, 1].set_xlabel("freq (cyc/sample)")
        # (c) drift spectrum (aggregate over layers)
        if drift_specs:
            fref = drift_specs[lmid][0]
            Pagg = np.zeros_like(drift_specs[lmid][1])
            for lid, (fd, Pd) in drift_specs.items():
                if len(Pd) == len(Pagg): Pagg += Pd / max(Pd.sum(), 1e-12)
            axes[1, 0].loglog(fref[1:], Pagg[1:], lw=0.9, color="darkorange")
            for s in SCALES:
                axes[1, 0].axvline(f_ricker[s], color="gray", ls=":", lw=0.6)
            axes[1, 0].axvline(f_train, color="red", ls="--", lw=1, label="f_train")
            axes[1, 0].axvspan(fref[1], f_train, color="red", alpha=0.08, label="drift band (coarse-only)")
            axes[1, 0].set_title("length-DRIFT spectrum  |FFT δ(Δ)|²  (Σ layers)"); axes[1, 0].legend(fontsize=7)
            axes[1, 0].set_xlabel("freq (cyc/sample)")
        # (d) low-freq fraction per layer
        axes[1, 1].bar(range(len(lowfrac)), lowfrac, color="steelblue")
        axes[1, 1].axhline(0.5, color="gray", ls=":")
        axes[1, 1].set_title("drift energy fraction below f_train, per layer")
        axes[1, 1].set_xlabel("layer"); axes[1, 1].set_ylabel("low-freq fraction")
        out = OUT / "spectral_coupling.png"
        fig.tight_layout(); fig.savefig(out, dpi=140)
        print(f"\nsaved: {out}")
    except Exception as e:
        print(f"(plot failed: {e})")


if __name__ == "__main__":
    main()

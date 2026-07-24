#!/usr/bin/env python3
"""
Is the 10^-3..10^-2 structure in the PaTH backbone spectrum the WAVELET's doing,
or PaTH-intrinsic (Householder eigen-angles)?

Compare per-layer E_base_raw (pure PaTH backbone logit) spectrum of:
  PRE  = 1r_baseline (the pre-finetune PaTH weights the QWAB models start from),
         loaded with a no_pe K=1 config shell (strict=False; wavelet params random
         but E_base_raw is pre-wavelet so unaffected) -> PaTH-only reference.
  QWAB = s42_delta_detach (K=8, post wavelet-finetuning).
Highlight/quantify the [1e-3, 1e-2] band. If the structure is present in PRE too
=> PaTH-intrinsic (not the wavelet). If only in QWAB => wavelet/finetune-induced.

Caveat: PRE is vanilla-PE-trained weights run at no_pe (the QWAB init point), so
this shows the backbone AT INITIALIZATION vs after finetuning; combined with the
scale-invariance result (cross-rho, K8-vs-K1) it isolates the wavelet's role.
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
QWAB = RUNS / "head_wise_scale_selection_vs_layer_wise/layer_wise/sigmoid_exp/s42_delta_detach/checkpoint-15000"
PRE_CKPT = RUNS / "1r_baseline_from_s/checkpoint-80000"
SHELL_CFG = RUNS / "pat225_scale_card/S1_me8_s42/checkpoint-15000"   # no_pe K=1 config shell for PRE
HOTPOT = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/hotpot_long_dev_uniform.jsonl"
OUT = Path("/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat234_scale_card")
L = 4096; DMAX = 4000; Q_LO = 512; N_SAMPLES = 4
BAND = (1e-3, 1e-2)


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


def backbone(model, tok, exs, dev):
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
                acc[lid][1:dm + 1] += en[:, i, i - np.arange(1, dm + 1)].mean(0); cnt[lid][1:dm + 1] += 1
    pa.PaTHAttention._build_ctxscale_shift_logit_bias_v0 = orig
    P, fgrid = {}, None
    for lid in layers:
        f, Pw = spectrum(acc[lid] / np.maximum(cnt[lid], 1))
        if f.size == 0: continue
        fgrid = f; P[lid] = Pw / max(Pw.sum(), 1e-12)
    return P, fgrid, layers


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

    # QWAB
    print("loading QWAB (K=8) ...", flush=True)
    cq = AutoConfig.from_pretrained(QWAB); cq.attn_implementation = "path_attn"
    cq.wavelet_logit_bias_log_every = 10 ** 9; cq.wavelet_ctxscale_eval_log_once = False
    mq = AutoModelForCausalLM.from_pretrained(QWAB, config=cq, torch_dtype=torch.float32).to(dev).eval()
    Pq, fg, layers = backbone(mq, tok, exs, dev)
    del mq; torch.cuda.empty_cache()

    # PRE = 1r_baseline weights into no_pe K=1 shell config
    print("loading PRE (1r_baseline @ no_pe shell) ...", flush=True)
    cp = AutoConfig.from_pretrained(SHELL_CFG); cp.attn_implementation = "path_attn"
    cp.pe_method = "no_pe"
    cp.wavelet_logit_bias_log_every = 10 ** 9; cp.wavelet_ctxscale_eval_log_once = False
    mp = AutoModelForCausalLM.from_pretrained(PRE_CKPT, config=cp, torch_dtype=torch.float32).to(dev).eval()
    Pp, fg2, _ = backbone(mp, tok, exs, dev)
    del mp; torch.cuda.empty_cache()

    bmask = (fg >= BAND[0]) & (fg <= BAND[1])
    print(f"\n=== band [{BAND[0]:.0e},{BAND[1]:.0e}] normalized power per layer ===")
    print(f"{'L':>3} | {'PRE(PaTH-only)':>14} | {'QWAB':>10} | {'QWAB/PRE':>9}")
    pr_b, qw_b = [], []
    for lid in layers:
        if lid not in Pp or lid not in Pq: continue
        pb = Pp[lid][bmask].sum(); qb = Pq[lid][bmask].sum()
        pr_b.append(pb); qw_b.append(qb)
        print(f"{lid:>3} | {pb:>14.4f} | {qb:>10.4f} | {qb/max(pb,1e-9):>9.2f}")
    print(f"\nAGG band power: PRE={np.mean(pr_b):.4f}  QWAB={np.mean(qw_b):.4f}  ratio={np.mean(qw_b)/max(np.mean(pr_b),1e-9):.2f}")
    print(">> ratio ~1 => structure PaTH-intrinsic (not wavelet); ratio >>1 => wavelet/finetune-induced")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 4, figsize=(19, 11))
        for ax, lid in zip(axes.flat, layers):
            if lid in Pp: ax.loglog(fg[fg > 0], Pp[lid][fg > 0], lw=0.8, color="gray", label="PRE PaTH-only")
            if lid in Pq: ax.loglog(fg[fg > 0], Pq[lid][fg > 0], lw=0.8, color="crimson", alpha=0.8, label="QWAB K=8")
            ax.axvspan(BAND[0], BAND[1], color="gold", alpha=0.15)
            ax.set_title(f"L{lid}", fontsize=8); ax.set_xlabel("f", fontsize=7)
            if lid == 0: ax.legend(fontsize=7)
        fig.suptitle("PaTH backbone spectrum: PRE (1r_baseline, no wavelet, no_pe) vs QWAB K=8 — band [1e-3,1e-2] shaded")
        out = OUT / "pathonly_vs_qwab_spectrum.png"
        fig.tight_layout(); fig.savefig(out, dpi=130)
        print(f"\nsaved: {out}")
    except Exception as e:
        print(f"(plot failed: {e})")


if __name__ == "__main__":
    main()

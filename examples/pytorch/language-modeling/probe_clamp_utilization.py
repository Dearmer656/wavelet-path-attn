#!/usr/bin/env python3
"""
How much do Clamp1 (per-scale p99) and Clamp2 (final ±g_bias_max) actually clip?
Does the fine-scale (rho1) ricker lose its important spike to Clamp1?

Uses the built-in _pat234_cap stage capture:
  S1_postnorm  = basis after per-scale RMS, BEFORE p99   (Clamp1 input)
  S2_postp99   = basis AFTER p99                          (Clamp1 output)
  S4pre_preclamp / S4post_postclamp = total bias before/after ±g_bias_max (Clamp2)

For each K=1 model (its single scale rho): report Clamp1 threshold vs peak, %clipped,
energy kept; Clamp2 saturation rate. Plot rho1 basis (S1 vs S2) over distance so the
spike-clipping is visible.
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
L = 512
G_BIAS_MAX = 4.0
MODELS = [("me0", 1, RUNS / "S1_me0_s42"), ("me8", 16, RUNS / "S1_me8_s42"),
          ("me16", 256, RUNS / "S1_me16_s42"), ("me28", 16384, RUNS / "S1_me28_s42")]
CKPT = "checkpoint-15000"


def render(q, ctx):
    c = "".join(f"Title: {t}\n{' '.join(s).strip()}\n\n" for t, s in ctx)
    return f"Context:\n{c}\nQuestion: {q}\nAnswer:"


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}")
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    ex = None
    with open(HOTPOT) as f:
        for line in f:
            e = json.loads(line)
            if e["meta"]["target_total_tokens"] == 4096: ex = e; break
    ids = tok(render(ex["question"], ex["context"]), return_tensors="pt", truncation=False)["input_ids"][:, :L].to(dev)

    print(f"=== Clamp utilization (L{L}, g_bias_max={G_BIAS_MAX}) ===")
    print(f"{'model':<7}{'rho':>7} | {'C1 thr':>8}{'C1 peak':>9}{'peak/thr':>9}{'%clip':>7}{'E_kept':>8} | {'C2 sat%':>8}{'C2 max|pre|':>11}")
    rho1_shape = None
    for label, rho, d in MODELS:
        ck = d / CKPT
        if not ck.exists(): print(f"{label:<7}{rho:>7} | (no ckpt)"); continue
        cfg = AutoConfig.from_pretrained(ck); cfg.attn_implementation = "path_attn"; cfg.path_attn_impl = "pytorch"
        cfg.wavelet_logit_bias_log_every = 10 ** 9; cfg.wavelet_ctxscale_eval_log_once = False
        model = AutoModelForCausalLM.from_pretrained(ck, config=cfg, torch_dtype=torch.float32).to(dev).eval()
        mods = [m for m in model.modules() if isinstance(m, pa.PaTHAttention)]
        for m in mods: m._pat234_cap = {}
        with torch.no_grad(): model(input_ids=ids)
        # gather across layers/chunks
        cap = {}
        for m in mods:
            for k, v in getattr(m, "_pat234_cap", {}).items():
                cap.setdefault(k, []).extend(v)
        # Clamp1: S1 (pre-p99) vs S2 (post-p99), matched by (lid,scale,q0)
        s1 = {(a, b, c): t for (a, b, c, t) in cap.get("S1_postnorm", [])}
        s2 = {(a, b, c): t for (a, b, c, t) in cap.get("S2_postp99", [])}
        thr_l, peak_l, clip_l, ek_l = [], [], [], []
        for key in s1:
            if key not in s2: continue
            pre = s1[key].numpy().ravel(); post = s2[key].numpy().ravel()
            thr = np.abs(post).max()
            peak = np.abs(pre).max()
            clip = float(np.mean(np.abs(pre) > thr * (1 - 1e-4)))
            ek = float((post ** 2).sum() / max((pre ** 2).sum(), 1e-12))
            thr_l.append(thr); peak_l.append(peak); clip_l.append(clip); ek_l.append(ek)
            if rho == 1 and rho1_shape is None and key[0] in (5, 6):   # a mid layer for rho1
                # one late query row over distance
                A = s1[key].numpy(); B = s2[key].numpy()  # [B,qc,T]
                qi = A.shape[1] - 1
                rho1_shape = (A[0, qi], B[0, qi])
        # Clamp2: S4pre vs S4post
        p4 = {(a, b): t for (a, b, t) in cap.get("S4pre_preclamp", [])}
        q4 = {(a, b): t for (a, b, t) in cap.get("S4post_postclamp", [])}
        sat_l, mx_l = [], []
        for key in p4:
            if key not in q4: continue
            pre = p4[key].numpy().ravel(); post = q4[key].numpy().ravel()
            sat_l.append(float(np.mean(np.abs(post) >= G_BIAS_MAX * (1 - 1e-4))))
            mx_l.append(float(np.abs(pre).max()))
        c1thr = np.mean(thr_l) if thr_l else float("nan")
        c1peak = np.mean(peak_l) if peak_l else float("nan")
        print(f"{label:<7}{rho:>7} | {c1thr:>8.3f}{c1peak:>9.3f}{c1peak/max(c1thr,1e-9):>9.2f}"
              f"{100*np.mean(clip_l) if clip_l else 0:>6.2f}%{np.mean(ek_l) if ek_l else 0:>8.3f} | "
              f"{100*np.mean(sat_l) if sat_l else 0:>7.3f}%{np.mean(mx_l) if mx_l else 0:>11.3f}")
        del model; torch.cuda.empty_cache()

    print("\nC1 thr=p99 clamp threshold; peak=max|basis| pre-clamp; peak/thr>1 => the spike is clipped;")
    print("%clip=fraction of entries clipped; E_kept=energy retained after p99. C2 sat%=positions hitting ±4.0.")

    if rho1_shape is not None:
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            A, B = rho1_shape; qi = len(A) - 1
            d = np.arange(len(A))
            fig, ax = plt.subplots(1, 2, figsize=(15, 5))
            ax[0].plot(A, label="S1 post-RMS (pre-p99)", lw=1); ax[0].plot(B, label="S2 post-p99", lw=1, ls="--")
            ax[0].set_title("rho1 basis over keys (one query row): does p99 clip the spike?")
            ax[0].set_xlabel("key index"); ax[0].legend(); ax[0].set_ylabel("basis value")
            lo = max(0, qi - 20)
            ax[1].plot(range(lo, len(A)), A[lo:], "o-", ms=3, label="pre-p99")
            ax[1].plot(range(lo, len(A)), B[lo:], "s--", ms=3, label="post-p99")
            ax[1].set_title("zoom near query (Δ small)"); ax[1].set_xlabel("key index"); ax[1].legend()
            out = OUT / "clamp_utilization_rho1.png"
            fig.tight_layout(); fig.savefig(out, dpi=140); print(f"saved: {out}")
        except Exception as e:
            print(f"(plot failed: {e})")


if __name__ == "__main__":
    main()

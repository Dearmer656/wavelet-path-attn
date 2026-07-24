#!/usr/bin/env python3
"""
After removing per-scale RMS (norm-off, raw ricker peak=1), how much does the
p99 clamp (Clamp1) still clip each scale? Compare norm-ON (RMS) vs norm-OFF (raw).

Hooks _maybe_clamp_p99 to capture (pre, post) for EVERY invocation, so it works
regardless of norm mode. Reports per scale: p99 threshold, pre-clamp peak, peak/thr,
%entries clipped, energy kept. Also final total-bias max (for Clamp2 relevance).
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
L = 512
G_BIAS_MAX = 4.0
MODELS = [("me0", 1, RUNS / "S1_me0_s42"), ("me8", 16, RUNS / "S1_me8_s42"),
          ("me16", 256, RUNS / "S1_me16_s42"), ("me28", 16384, RUNS / "S1_me28_s42")]
CKPT = "checkpoint-15000"


def render(q, ctx):
    c = "".join(f"Title: {t}\n{' '.join(s).strip()}\n\n" for t, s in ctx)
    return f"Context:\n{c}\nQuestion: {q}\nAnswer:"


def run_one(ck, tok, ids, dev, norm_off):
    cfg = AutoConfig.from_pretrained(ck); cfg.attn_implementation = "path_attn"; cfg.path_attn_impl = "pytorch"
    cfg.wavelet_logit_bias_norm_disable = bool(norm_off)
    cfg.wavelet_logit_bias_log_every = 10 ** 9; cfg.wavelet_ctxscale_eval_log_once = False
    model = AutoModelForCausalLM.from_pretrained(ck, config=cfg, torch_dtype=torch.float32).to(dev).eval()
    for m in model.modules():
        if isinstance(m, pa.PaTHAttention):
            m.wavelet_logit_bias_norm_disable = bool(norm_off)   # runtime attr too
    caps = []
    orig = pa.PaTHAttention._maybe_clamp_p99
    def hooked(self, x):
        out = orig(self, x)
        caps.append((x.detach().float().cpu().numpy().ravel(), out.detach().float().cpu().numpy().ravel()))
        return out
    pa.PaTHAttention._maybe_clamp_p99 = hooked
    with torch.no_grad(): model(input_ids=ids)
    pa.PaTHAttention._maybe_clamp_p99 = orig
    del model; torch.cuda.empty_cache()
    thr, peak, clip, ek = [], [], [], []
    for pre, post in caps:
        t = np.abs(post).max(); p = np.abs(pre).max()
        thr.append(t); peak.append(p)
        clip.append(float(np.mean(np.abs(pre) > t * (1 - 1e-4))))
        ek.append(float((post ** 2).sum() / max((pre ** 2).sum(), 1e-12)))
    return (np.mean(thr), np.mean(peak), np.mean(clip), np.mean(ek)) if thr else (np.nan,) * 4


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

    print(f"\n=== Clamp1 (p99) impact: norm-ON (RMS) vs norm-OFF (raw ricker), L{L} ===")
    print(f"{'ρ':>7} | {'mode':>8} | {'thr':>8}{'peak':>9}{'peak/thr':>9}{'%clip':>7}{'E_kept':>8}")
    for label, rho, d in MODELS:
        ck = d / CKPT
        if not ck.exists(): print(f"{rho:>7} | (no ckpt)"); continue
        for tag, noff in [("RMS-on", False), ("norm-off", True)]:
            thr, peak, clip, ek = run_one(ck, tok, ids, dev, noff)
            print(f"{rho:>7} | {tag:>8} | {thr:>8.3f}{peak:>9.3f}{peak/max(thr,1e-9):>9.2f}{100*clip:>6.2f}%{ek:>8.3f}")
        print("  " + "-" * 50)
    print("\npeak/thr>1 => spike clipped; E_kept=energy retained after p99.")
    print("norm-off raw ricker peak≈1; if E_kept still low => p99 clips even the raw fine spike.")


if __name__ == "__main__":
    main()

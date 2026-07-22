#!/usr/bin/env python3
"""
PAT-234 preliminary check (no training). Two questions:

Q1 (empirical): is the query amplitude gate g_{i,0} preserved in the final
    effective wavelet bias, or cancelled by normalization? Measure
    Corr_i( g_{i,0}, RMS_j(eff_bias[i,:]) ). POSITIVE => g_{i,0} survives
    => the "RMS-norm cancels the amplitude gate" premise is refuted.

Q2 (analytical, no checkpoint): does the current per-scale basis RMS-norm
    (applied to the RAW, uncentered Ricker) leave coarse scales as near-constant
    (softmax-INVISIBLE) contributions? For each scale, after rms-norm over keys,
    visible_fraction = RMS(basis - mean_k basis) / RMS(basis) = sqrt(1 - mean^2).
    Coarse (near-constant) scales -> visible_fraction ~ 0.  This is what the
    proposed CENTERING step (u^c = u - mean_k u) would fix.
"""
import json, sys, math
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
TARGET_LEN = 2048   # memory-safe; mechanism is length-independent
N_SAMPLES = 8


def ricker(u):
    return (1.0 - u**2) * np.exp(-0.5 * u**2)


def q2_analytical():
    print("=== Q2 (analytical): per-scale softmax-visible fraction after current RMS-norm ===")
    print("  scale rho | mean(basis_norm) | visible_frac = sqrt(1-mean^2)  [~0 => softmax-INVISIBLE]")
    K8_scales = [2.0**e for e in range(0, 15, 2)]   # production K=8 grid: 1,4,...,16384
    for L in (512, 2048, 4096):
        print(f"  -- L={L} --")
        tau = np.arange(0, L, dtype=np.float64)
        for rho in K8_scales:
            b = ricker(tau / rho)
            rms = math.sqrt((b**2).mean())
            b_norm = b / (rms + 1e-6)
            m = b_norm.mean()
            vis = math.sqrt(max(0.0, (b_norm**2).mean() - m**2))
            print(f"     rho={rho:>7.0f} | mean={m:>8.4f} | visible_frac={vis:.4f}")


def render_input(q, ctx):
    c = "".join(f"Title: {t}\n{' '.join(s).strip()}\n\n" for t, s in ctx)
    return f"Context:\n{c}\nQuestion: {q}\nAnswer:"


def q1_empirical():
    print("\n=== Q1 (empirical): does g_{i,0} survive into the effective bias? ===")
    device = torch.device("cuda")
    cfg = AutoConfig.from_pretrained(CKPT)
    cfg.attn_implementation = "path_attn"
    cfg.wavelet_logit_bias_log_every = 10**9
    cfg.wavelet_ctxscale_eval_log_once = False
    model = AutoModelForCausalLM.from_pretrained(CKPT, config=cfg, torch_dtype=torch.float32).to(device).eval()

    # monkeypatch the active bias builder to stash its returned bias + g0
    orig = pa.PaTHAttention._build_ctxscale_shift_logit_bias_v0
    store = {}
    def wrapped(self, **kw):
        out = orig(self, **kw)
        try:
            eff = out[0].detach().float()            # [B,H,T,T] wavelet logit bias
            nm = getattr(self, "_last_ctxscale_null_mass", None)   # pi_null
            lid = int(getattr(self, "layer_idx", -1) or -1)
            store[lid] = (eff, nm.detach().float() if torch.is_tensor(nm) else None)
        except Exception as e:
            store["err"] = str(e)
        return out
    pa.PaTHAttention._build_ctxscale_shift_logit_bias_v0 = wrapped

    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    exs = []
    with open(HOTPOT) as f:
        for line in f:
            ex = json.loads(line)
            if ex["meta"]["target_total_tokens"] == TARGET_LEN:
                exs.append(ex)
            if len(exs) >= N_SAMPLES: break

    corrs = {}   # layer -> list of per-example correlations
    for ex in exs:
        store.clear()
        ids = tok(render_input(ex["question"], ex["context"]), return_tensors="pt", truncation=False)["input_ids"].to(device)
        with torch.no_grad():
            model(input_ids=ids)
        for lid, (eff, nm) in list(store.items()):
            if not isinstance(lid, int) or nm is None: continue
            # per-query bias RMS over valid (causal, lower-tri) keys, mean over heads/batch
            e = eff.squeeze(0)                       # [H,T,T] or [T,T]
            if e.dim() == 2: e = e.unsqueeze(0)
            T = e.shape[-1]
            mask = torch.tril(torch.ones(T, T, device=e.device))  # valid keys j<=i
            cnt = mask.sum(-1).clamp_min(1.0)
            # CENTERED (softmax-relevant) bias RMS: subtract per-query mean over valid keys
            mean_k = (e * mask).sum(-1) / cnt                      # [H,T]
            e_c = (e - mean_k.unsqueeze(-1)) * mask
            bias_rms = torch.sqrt(((e_c**2).sum(-1) / cnt).clamp_min(0)).mean(0)  # [T] centered, mean over heads
            # g0 = 1 - pi_null (per query), mean over heads if headwise
            g0 = (1.0 - nm.squeeze(0))
            while g0.dim() > 1: g0 = g0.mean(dim=-1) if g0.shape[-1] > 1 else g0.squeeze(-1)
            g0 = g0.reshape(-1)[:T]
            br = bias_rms.reshape(-1)[:T].cpu().numpy()
            g0n = g0.cpu().numpy()
            # drop first few positions (degenerate causal window)
            sl = slice(8, T)
            if br[sl].std() > 1e-9 and g0n[sl].std() > 1e-9:
                corrs.setdefault(lid, []).append(float(np.corrcoef(g0n[sl], br[sl])[0, 1]))

    print("  layer | mean Corr(g0, per-query CENTERED bias RMS) over examples  [>0 => g_{i,0} survives]")
    all_c = []
    for lid in sorted(corrs):
        c = float(np.mean(corrs[lid])); all_c.append(c)
        print(f"    L{lid:>2} | {c:+.4f}")
    if all_c:
        print(f"  AGGREGATE mean corr = {np.mean(all_c):+.4f}")
        print("  >> POSITIVE => amplitude gate g_{i,0} is preserved (cancellation hypothesis REFUTED)")
        print("  >> ~0       => g_{i,0} does not reach the effective bias (cancellation supported)")


if __name__ == "__main__":
    q2_analytical()
    q1_empirical()

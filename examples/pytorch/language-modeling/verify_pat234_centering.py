#!/usr/bin/env python3
"""
PAT-234 positive control: verify the centering code actually modifies the
intermediate effective bias correctly. Same model + same input, toggle
wavelet_logit_bias_center on/off, compare per-layer:
  - M_eff  = CENTERED (softmax-relevant) bias RMS  -> should INCREASE with centering
  - raw    = raw bias RMS                          -> DC removed, may decrease
If M_eff(on) > M_eff(off), coarse scales that were softmax-invisible DC now
contribute visible variation => centering works as intended.
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

CKPT = "/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat225_scale_card/S1_me28_s42/checkpoint-15000"
HOTPOT = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/hotpot_long_dev_uniform.jsonl"
TARGET_LEN = 512   # training length: where coarse scales are most invisible (rho>=1024)
N_SAMPLES = 4


def render(q, ctx):
    c = "".join(f"Title: {t}\n{' '.join(s).strip()}\n\n" for t, s in ctx)
    return f"Context:\n{c}\nQuestion: {q}\nAnswer:"


def main():
    dev = torch.device("cuda")
    cfg = AutoConfig.from_pretrained(CKPT)
    cfg.attn_implementation = "path_attn"
    cfg.wavelet_logit_bias_log_every = 10**9
    cfg.wavelet_ctxscale_eval_log_once = False
    model = AutoModelForCausalLM.from_pretrained(CKPT, config=cfg, torch_dtype=torch.float32).to(dev).eval()

    orig = pa.PaTHAttention._build_ctxscale_shift_logit_bias_v0
    store = {}
    def wrapped(self, **kw):
        out = orig(self, **kw)
        try:
            eff = out[0].detach().float()
            base = kw.get("E_base_raw", None)
            if base is not None:               # out[0] = base PaTH logits + wavelet; isolate wavelet-only
                eff = eff - base.detach().float()
            store[int(getattr(self, "layer_idx", -1) or -1)] = eff
        except Exception:
            pass
        return out
    pa.PaTHAttention._build_ctxscale_shift_logit_bias_v0 = wrapped

    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    exs = []
    with open(HOTPOT) as f:
        for line in f:
            ex = json.loads(line)
            if ex["meta"]["target_total_tokens"] == 4096:
                exs.append(ex)
            if len(exs) >= N_SAMPLES: break

    def measure(center_on):
        # set the flag on every attention core
        for m in model.modules():
            if isinstance(m, pa.PaTHAttention):
                m.wavelet_logit_bias_center = bool(center_on)
        acc_c = {}; acc_r = {}
        for ex in exs:
            store.clear()
            ids = tok(render(ex["question"], ex["context"]), return_tensors="pt", truncation=False)["input_ids"][:, :TARGET_LEN].to(dev)
            with torch.no_grad():
                model(input_ids=ids)
            for lid, eff in store.items():
                e = eff.squeeze(0)
                if e.dim() == 2: e = e.unsqueeze(0)
                T = e.shape[-1]
                mask = torch.tril(torch.ones(T, T, device=e.device))
                cnt = mask.sum(-1).clamp_min(1.0)
                raw = torch.sqrt(((e**2 * mask).sum(-1) / cnt).clamp_min(0)).mean().item()
                mk = (e * mask).sum(-1) / cnt
                ec = (e - mk.unsqueeze(-1)) * mask
                cen = torch.sqrt(((ec**2).sum(-1) / cnt).clamp_min(0)).mean().item()
                acc_c.setdefault(lid, []).append(cen)
                acc_r.setdefault(lid, []).append(raw)
        return ({l: float(np.mean(v)) for l, v in acc_c.items()},
                {l: float(np.mean(v)) for l, v in acc_r.items()})

    c_off, r_off = measure(False)
    c_on, r_on = measure(True)

    print("=== PAT-234 centering positive control (same model, flag off vs on) ===")
    print("  layer | M_eff(off) | M_eff(on) |  ratio  || raw(off) | raw(on)")
    L = sorted(c_off)
    for lid in L:
        co, cn = c_off[lid], c_on.get(lid, float('nan'))
        ro, rn = r_off[lid], r_on.get(lid, float('nan'))
        print(f"   L{lid:>2}  | {co:.5f}   | {cn:.5f}  | {cn/max(co,1e-9):>5.2f}x || {ro:.4f}  | {rn:.4f}")
    mo = float(np.mean([c_off[l] for l in L])); mn = float(np.mean([c_on[l] for l in L]))
    print(f"  AGG M_eff: off={mo:.5f}  on={mn:.5f}  ratio={mn/max(mo,1e-9):.2f}x")
    print(">> on > off  => centering makes coarse scales softmax-VISIBLE (code works as intended)")
    print(">> on ~= off => centering did NOT change the effective bias (BUG - do not trust C training)")


if __name__ == "__main__":
    main()

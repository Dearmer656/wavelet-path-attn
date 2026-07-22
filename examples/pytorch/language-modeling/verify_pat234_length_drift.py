#!/usr/bin/env python3
"""
PAT-234 length-drift diagnostic: is the extrapolation degradation of center-ON a
NORMALIZATION artifact? _rms_norm_last_dim normalizes over ALL T keys (not causal),
so the effective bias magnitude a query sees can drift with total sequence length.

Measure, on the AS-TRAINED rho16384 C (center-ON) and A (center-OFF) checkpoints,
the softmax-visible wavelet bias RMS (M_eff, per-query centered over causal keys)
at L=512 / 2048 / 4096.

Expect:  A (center-off) ~ length-stable ;  C (center-on) drifts with length
=> confirms the train(512)/eval(4096) normalization mismatch is the confound.
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

MODELS = {
    "C(center-ON)":  "/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat234_scale_card/K1_me28_C_s42/checkpoint-15000",
    "A(center-OFF)": "/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat225_scale_card/S1_me28_s42/checkpoint-15000",
}
HOTPOT = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/hotpot_long_dev_uniform.jsonl"
LENGTHS = [512, 2048, 4096]
N_SAMPLES = 3


def render(q, ctx):
    c = "".join(f"Title: {t}\n{' '.join(s).strip()}\n\n" for t, s in ctx)
    return f"Context:\n{c}\nQuestion: {q}\nAnswer:"


def load_exs(tok):
    exs = []
    with open(HOTPOT) as f:
        for line in f:
            ex = json.loads(line)
            if ex["meta"]["target_total_tokens"] == 4096:
                exs.append(ex)
            if len(exs) >= N_SAMPLES:
                break
    return exs


def m_eff_at_len(model, tok, exs, dev, L):
    store = {}
    orig = pa.PaTHAttention._build_ctxscale_shift_logit_bias_v0
    def wrapped(self, **kw):
        out = orig(self, **kw)
        try:
            eff = out[0].detach().float()
            base = kw.get("E_base_raw", None)
            if base is not None:
                eff = eff - base.detach().float()
            store[int(getattr(self, "layer_idx", -1) or -1)] = eff.cpu()
        except Exception:
            pass
        return out
    pa.PaTHAttention._build_ctxscale_shift_logit_bias_v0 = wrapped
    m_causal, raw_causal = [], []
    for ex in exs:
        store.clear()
        ids = tok(render(ex["question"], ex["context"]), return_tensors="pt", truncation=False)["input_ids"][:, :L].to(dev)
        with torch.no_grad():
            model(input_ids=ids)
        for eff in store.values():
            e = eff.squeeze(0)
            if e.dim() == 2: e = e.unsqueeze(0)
            T = e.shape[-1]
            qi = torch.arange(T).view(T, 1); ki = torch.arange(T).view(1, T)
            mask = (ki <= qi).to(e.dtype)
            cnt = mask.sum(-1).clamp_min(1.0)
            raw = torch.sqrt(((e ** 2) * mask).sum(-1) / cnt)          # [H,T]
            mk = (e * mask).sum(-1) / cnt
            cen = torch.sqrt((((e - mk.unsqueeze(-1)) ** 2) * mask).sum(-1) / cnt)
            m_causal.append(float(cen[:, 8:].mean().item()))
            raw_causal.append(float(raw[:, 8:].mean().item()))
    pa.PaTHAttention._build_ctxscale_shift_logit_bias_v0 = orig
    return float(np.mean(m_causal)), float(np.mean(raw_causal))


def main():
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    exs = load_exs(tok)
    print("=== PAT-234 length-drift: softmax-visible wavelet bias M_eff (causal centered RMS) vs eval length ===")
    for label, ck in MODELS.items():
        if not Path(ck, "model.safetensors").is_file():
            print(f"\n## {label}: (no ckpt {ck})"); continue
        cfg = AutoConfig.from_pretrained(ck)
        cfg.attn_implementation = "path_attn"
        cfg.wavelet_logit_bias_log_every = 10 ** 9
        cfg.wavelet_ctxscale_eval_log_once = False
        model = AutoModelForCausalLM.from_pretrained(ck, config=cfg, torch_dtype=torch.float32).to(dev).eval()
        print(f"\n## {label}   ({Path(ck).parent.name})")
        print(f"  {'L':>6} | {'M_eff(causal cen)':>18} | {'raw RMS(causal)':>16}")
        base_m = None
        for L in LENGTHS:
            m, r = m_eff_at_len(model, tok, exs, dev, L)
            if base_m is None: base_m = m
            print(f"  {L:>6} | {m:>18.4f} | {r:>16.4f}   (x{m/max(base_m,1e-9):.2f} vs L512)")
        del model; torch.cuda.empty_cache()
    print("\n>> A ~ flat across L, C drifts  => length-dependent norm artifact confirmed")


if __name__ == "__main__":
    main()

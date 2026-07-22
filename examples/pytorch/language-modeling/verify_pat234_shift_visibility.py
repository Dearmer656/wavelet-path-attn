#!/usr/bin/env python3
"""
PAT-234 corrected-Q2: does the learned scale-dependent SHIFT provide the
softmax-visibility that the (unshifted) analytical Q2 said coarse scales lack?

For each K=1 scale model, measure the CENTERED (softmax-relevant) effective
bias RMS = M_eff with the shift ON (default) vs OFF. If coarse scales show
large M_eff with shift-on but ~0 with shift-off, the shift (not the raw Ricker)
is what makes them visible -> Q2's "coarse invisible" premise (which assumed a
fixed-center Ricker) is refuted, and centering has nothing to fix.
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

SC = "/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat225_scale_card"
MODELS = {
    "rho1": "S1_me0_s42", "rho16": "S1_me8_s42", "rho128": "S1_s42",
    "rho1024": "S1_me20_s42", "rho16384": "S1_me28_s42",
}
HOTPOT = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/hotpot_long_dev_uniform.jsonl"
TARGET_LEN = 512
N_SAMPLES = 4


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


def m_eff(model, tok, exs, dev, shift_on):
    for m in model.modules():
        if isinstance(m, pa.PaTHAttention):
            m.wavelet_ctxscale_scale_dependent_shift = bool(shift_on)
    store = {}
    orig = pa.PaTHAttention._build_ctxscale_shift_logit_bias_v0
    def wrapped(self, **kw):
        out = orig(self, **kw)
        try:
            store[int(getattr(self, "layer_idx", -1) or -1)] = out[0].detach().float()
        except Exception:
            pass
        return out
    pa.PaTHAttention._build_ctxscale_shift_logit_bias_v0 = wrapped
    vals = []
    for ex in exs:
        store.clear()
        ids = tok(render(ex["question"], ex["context"]), return_tensors="pt", truncation=False)["input_ids"][:, :TARGET_LEN].to(dev)
        with torch.no_grad():
            model(input_ids=ids)
        for eff in store.values():
            e = eff.squeeze(0)
            if e.dim() == 2: e = e.unsqueeze(0)
            T = e.shape[-1]
            mask = torch.tril(torch.ones(T, T, device=e.device))
            cnt = mask.sum(-1).clamp_min(1.0)
            mk = (e * mask).sum(-1) / cnt
            ec = (e - mk.unsqueeze(-1)) * mask
            vals.append(torch.sqrt(((ec**2).sum(-1) / cnt).clamp_min(0)).mean().item())
    pa.PaTHAttention._build_ctxscale_shift_logit_bias_v0 = orig
    return float(np.mean(vals)) if vals else float("nan")


def main():
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    exs = load_exs(tok)
    print("=== corrected-Q2: M_eff (softmax-visible bias RMS) with learned SHIFT on vs off, L=512 ===")
    print("  scale     | M_eff(shift ON) | M_eff(shift OFF) | ratio ON/OFF")
    for label, run in MODELS.items():
        ck = f"{SC}/{run}/checkpoint-15000"
        if not Path(ck).is_dir():
            print(f"  {label:<9} | (no ckpt {run})"); continue
        cfg = AutoConfig.from_pretrained(ck)
        cfg.attn_implementation = "path_attn"
        cfg.wavelet_logit_bias_log_every = 10**9
        cfg.wavelet_ctxscale_eval_log_once = False
        model = AutoModelForCausalLM.from_pretrained(ck, config=cfg, torch_dtype=torch.float32).to(dev).eval()
        on = m_eff(model, tok, exs, dev, True)
        off = m_eff(model, tok, exs, dev, False)
        print(f"  {label:<9} | {on:>13.4f}  | {off:>15.4f}  | {on/max(off,1e-9):>6.2f}x", flush=True)
        del model; torch.cuda.empty_cache()
    print(">> coarse scales: large ON, ~0 OFF => the SHIFT provides visibility (Q2 unshifted premise refuted)")


if __name__ == "__main__":
    main()

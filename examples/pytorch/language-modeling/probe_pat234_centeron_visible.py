#!/usr/bin/env python3
"""
PAT-234: measure the AS-TRAINED softmax-visible wavelet bias RMS (M_eff) per
scale for the center-ON K=1 sweep checkpoints, and compare to the center-OFF
counterparts. Confirms directly on the trained models that centering (a) lifted
coarse scales and (b) equalized the visible value range across scales.

Usage: probe_pat234_centeron_visible.py [STEP]   (default STEP=15000)
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

STEP = int(sys.argv[1]) if len(sys.argv) > 1 else 15000
ON  = "/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat234_scale_card"
OFF = "/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat225_scale_card"
# label -> (center-ON run, center-OFF run)
MODELS = {
    "rho1":     ("K1_me0_C_s42",  "S1_me0_s42"),
    "rho16":    ("K1_me8_C_s42",  "S1_me8_s42"),
    "rho128":   ("K1_me14_C_s42", "S1_s42"),
    "rho1024":  ("K1_me20_C_s42", "S1_me20_s42"),
    "rho16384": ("K1_me28_C_s42", "S1_me28_s42"),
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


def m_eff_of(ckpt, tok, exs, dev):
    if not Path(ckpt, "model.safetensors").is_file():
        return float("nan")
    cfg = AutoConfig.from_pretrained(ckpt)
    cfg.attn_implementation = "path_attn"
    cfg.wavelet_logit_bias_log_every = 10**9
    cfg.wavelet_ctxscale_eval_log_once = False
    model = AutoModelForCausalLM.from_pretrained(ckpt, config=cfg, torch_dtype=torch.float32).to(dev).eval()
    store = {}
    orig = pa.PaTHAttention._build_ctxscale_shift_logit_bias_v0
    def wrapped(self, **kw):
        out = orig(self, **kw)
        try:
            eff = out[0].detach().float()
            base = kw.get("E_base_raw", None)
            if base is not None:
                eff = eff - base.detach().float()
            store[int(getattr(self, "layer_idx", -1) or -1)] = eff
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
    del model; torch.cuda.empty_cache()
    return float(np.mean(vals)) if vals else float("nan")


def main():
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    exs = load_exs(tok)
    print(f"=== AS-TRAINED softmax-visible M_eff per scale, center-ON vs center-OFF (step {STEP}, L=512) ===")
    print("  scale     | M_eff(center-ON) | M_eff(center-OFF) | ON/OFF")
    ons = []
    for label, (on_run, off_run) in MODELS.items():
        on  = m_eff_of(f"{ON}/{on_run}/checkpoint-{STEP}", tok, exs, dev)
        off = m_eff_of(f"{OFF}/{off_run}/checkpoint-15000", tok, exs, dev)
        ons.append(on)
        print(f"  {label:<9} | {on:>14.4f}   | {off:>15.4f}   | {on/max(off,1e-9):>5.2f}x", flush=True)
    fin = [v for v in ons if v == v]
    if fin:
        print(f">> center-ON spread: min={min(fin):.4f} max={max(fin):.4f} ratio={max(fin)/max(min(fin),1e-9):.2f}x  (near 1 => value range equalized)")


if __name__ == "__main__":
    main()

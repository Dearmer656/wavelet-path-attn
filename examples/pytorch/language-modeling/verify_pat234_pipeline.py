#!/usr/bin/env python3
"""
PAT-234 rigorous pipeline probe (addresses reviewer requirements).

For each K=1 scale model, and for center in {OFF,ON} x shift in {ON,OFF},
measure the softmax-VISIBLE (per-query centered) RMS and raw RMS of the wavelet
bias at every stage of the assembly, plus the actual attention KL it induces:

  stage S1  = basis after RMS-norm            (pre-gain, pre-clamp)  <- centering shape effect
  stage S2  = after p99 clamp                 (pre-gain)
  stage S3  = after x pi_scale (gain g0.w)    (post-gain, pre-final-clamp)
  stage S4pre  = after x g_layer              (pre final g_bias clamp)
  stage S4post = after clamp(+-g_bias_max)    (post-clamp) == final wavelet bias
  attn_KL   = KL( softmax(base+eff) || softmax(base) ) over causal keys

Explicit comparisons the reviewer asked for:
  - uncentered->RMS vs centered->RMS : S1 centered-RMS with center OFF vs ON
  - pre/post-gain                    : S1 vs S3
  - pre/post-clamp                   : S4pre vs S4post
  - shift ON/OFF per scale           : outer toggle
  - attention KL                     : final, per condition

Runs on EXISTING center-off checkpoints (toggles are eval-time); no training.
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

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


def stage_rms(e):
    """e: [B,Q,T] or [Q,T]. Return (raw_rms, centered_rms) over causal keys, mean over queries."""
    if e.dim() == 3:
        e = e[0]
    Q, T = e.shape[-2], e.shape[-1]
    q_idx = torch.arange(Q).view(Q, 1)
    k_idx = torch.arange(T).view(1, T)
    mask = (k_idx <= q_idx).to(e.dtype)
    cnt = mask.sum(-1).clamp_min(1.0)
    raw = torch.sqrt(((e ** 2) * mask).sum(-1) / cnt)
    mk = (e * mask).sum(-1) / cnt
    cen = torch.sqrt((((e - mk.unsqueeze(-1)) ** 2) * mask).sum(-1) / cnt)
    # drop first 8 degenerate query rows
    sl = slice(8, Q)
    return float(raw[sl].mean().item()), float(cen[sl].mean().item())


def attn_kl(full, base):
    """full/base: [H,T,T] or [T,T] logits. KL(softmax(full)||softmax(base)) over causal keys, mean."""
    if full.dim() == 2:
        full = full.unsqueeze(0); base = base.unsqueeze(0)
    H, T, _ = full.shape
    q_idx = torch.arange(T).view(T, 1)
    k_idx = torch.arange(T).view(1, T)
    causal = (k_idx <= q_idx)
    neg = torch.finfo(full.dtype).min
    lf = full.masked_fill(~causal, neg)
    lb = base.masked_fill(~causal, neg)
    pf = torch.softmax(lf, dim=-1)
    logpf = torch.log_softmax(lf, dim=-1)
    logpb = torch.log_softmax(lb, dim=-1)
    kl = (pf * (logpf - logpb)).sum(-1)  # [H,T]
    kl = kl[:, 8:]
    return float(kl.mean().item())


def agg_stage(cap, key):
    """Mean stage_rms over all (layer) entries for a stage key. Returns (raw, cen) or (nan,nan)."""
    ents = cap.get(key, [])
    if not ents:
        return float("nan"), float("nan")
    raws, cens = [], []
    for ent in ents:
        e = ent[-1]  # tensor is last element
        r, c = stage_rms(e)
        raws.append(r); cens.append(c)
    return float(np.mean(raws)), float(np.mean(cens))


def run_condition(model, tok, exs, dev, center_on, shift_on):
    shared_stage = {}
    shared_logits = {}
    for m in model.modules():
        if isinstance(m, pa.PaTHAttention):
            m.wavelet_logit_bias_center = bool(center_on)
            m.wavelet_ctxscale_scale_dependent_shift = bool(shift_on)
            m._pat234_cap = shared_stage
    orig = pa.PaTHAttention._build_ctxscale_shift_logit_bias_v0
    def wrapped(self, **kw):
        out = orig(self, **kw)
        try:
            lid = int(getattr(self, "layer_idx", -1) or -1)
            base = kw.get("E_base_raw", None)
            full = out[0].detach().float()
            if base is not None:
                shared_logits[lid] = (full, base.detach().float())
        except Exception:
            pass
        return out
    pa.PaTHAttention._build_ctxscale_shift_logit_bias_v0 = wrapped

    acc = {k: [] for k in ["S1", "S2", "S3", "S4pre", "S4post", "KL"]}
    for ex in exs:
        shared_stage.clear(); shared_logits.clear()
        ids = tok(render(ex["question"], ex["context"]), return_tensors="pt", truncation=False)["input_ids"][:, :TARGET_LEN].to(dev)
        with torch.no_grad():
            model(input_ids=ids)
        acc["S1"].append(agg_stage(shared_stage, "S1_postnorm")[1])       # centered
        acc["S2"].append(agg_stage(shared_stage, "S2_postp99")[1])
        acc["S3"].append(agg_stage(shared_stage, "S3_postgain")[1])
        acc["S4pre"].append(agg_stage(shared_stage, "S4pre_preclamp")[1])
        acc["S4post"].append(agg_stage(shared_stage, "S4post_postclamp")[1])
        kls = []
        for lid, (full, base) in shared_logits.items():
            f2 = full[0] if full.dim() == 4 else full
            b2 = base[0] if base.dim() == 4 else base
            kls.append(attn_kl(f2, b2))
        acc["KL"].append(float(np.mean(kls)) if kls else float("nan"))

    pa.PaTHAttention._build_ctxscale_shift_logit_bias_v0 = orig
    for m in model.modules():
        if isinstance(m, pa.PaTHAttention):
            m._pat234_cap = None
    return {k: float(np.nanmean(v)) for k, v in acc.items()}


def main():
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    exs = load_exs(tok)
    print("=== PAT-234 pipeline probe: softmax-VISIBLE (per-query centered) RMS per stage + attention KL ===")
    print("stages: S1=post-norm  S3=post-gain  S4pre=pre g_bias clamp  S4post=final ; KL=attn KL(full||base)")
    for label, run in MODELS.items():
        ck = f"{SC}/{run}/checkpoint-15000"
        if not Path(ck, "model.safetensors").is_file():
            print(f"\n## {label}: (no ckpt {run})"); continue
        cfg = AutoConfig.from_pretrained(ck)
        cfg.attn_implementation = "path_attn"
        cfg.wavelet_logit_bias_log_every = 10 ** 9
        cfg.wavelet_ctxscale_eval_log_once = False
        cfg.wavelet_ctxscale_chunk_q = TARGET_LEN  # single chunk -> clean [T,T] capture
        model = AutoModelForCausalLM.from_pretrained(ck, config=cfg, torch_dtype=torch.float32).to(dev).eval()
        print(f"\n## {label} ({run})")
        print(f"{'center':>6} {'shift':>5} | {'S1_cen':>8} {'S3_cen':>8} {'S4pre':>8} {'S4post':>8} | {'attn_KL':>10}")
        res = {}
        for center_on in (False, True):
            for shift_on in (True, False):
                r = run_condition(model, tok, exs, dev, center_on, shift_on)
                res[(center_on, shift_on)] = r
                print(f"{str(center_on):>6} {str(shift_on):>5} | {r['S1']:>8.4f} {r['S3']:>8.4f} {r['S4pre']:>8.4f} {r['S4post']:>8.4f} | {r['KL']:>10.3e}")
        # explicit contrasts (shift ON)
        off = res[(False, True)]; on = res[(True, True)]
        print(f"  -> uncentered->RMS vs centered->RMS (S1): {off['S1']:.4f} -> {on['S1']:.4f}  (x{on['S1']/max(off['S1'],1e-9):.2f})")
        print(f"  -> pre/post-gain (S1->S3, center ON):     {on['S1']:.4f} -> {on['S3']:.4f}")
        print(f"  -> pre/post-clamp (S4pre->S4post, ON):    {on['S4pre']:.4f} -> {on['S4post']:.4f}")
        print(f"  -> attn KL: centerOFF={off['KL']:.3e}  centerON={on['KL']:.3e}  (x{on['KL']/max(off['KL'],1e-12):.2f})")
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

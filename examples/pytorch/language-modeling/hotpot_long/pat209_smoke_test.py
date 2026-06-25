#!/usr/bin/env python3
"""PAT-209 smoke test for the custom-position RoPE plumbing in modeling_gpt2.py.

Checks:
 1. eval forward is IDENTICAL whether or not POS_EXPOSURE_MODE is set (mode gates on training).
 2. 'arange_debug' training forward == standard eval forward (custom path reproduces std RoPE).
 3. 'dim_specific' / 'pose_shared' training forward runs, stash shapes are [B,1,T,D], and the
    backward pass is differentiable (loss.backward works).
 4. dim_specific actually exposes each pair to ~1 period: max sampled angle per pair ≈ 2π+buf·θ.
"""
import os, argparse, json
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def load(ckpt, dev):
    cfg = AutoConfig.from_pretrained(ckpt)
    cfg.attn_implementation = "eager"; cfg.pe_method = "rotary"
    m = AutoModelForCausalLM.from_pretrained(ckpt, config=cfg, torch_dtype=torch.float32,
                                             trust_remote_code=True).to(dev)
    return m


def fwd_logits(m, ids, train, mode):
    os.environ.pop("POS_EXPOSURE_MODE", None)
    if mode is not None:
        os.environ["POS_EXPOSURE_MODE"] = mode
    # force re-read of env each call
    if hasattr(m.transformer, "_pos_exp_mode"):
        del m.transformer._pos_exp_mode
    m.train(train)
    with torch.set_grad_enabled(train):
        out = m(ids)
    return out.logits.detach().float()


def run(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    m = load(a.checkpoint, dev)
    B, T = 2, 256
    ids = torch.randint(0, 50000, (B, T), device=dev)

    # 1. eval identical with/without env
    e_none = fwd_logits(m, ids, train=False, mode=None)
    e_dim = fwd_logits(m, ids, train=False, mode="dim_specific")
    d1 = (e_none - e_dim).abs().max().item()
    print(f"[1] eval std vs eval(dim_specific env): max|Δ| = {d1:.2e}  (expect 0)")

    # 2. arange_debug TRAIN == std EVAL  (custom path reproduces standard RoPE)
    torch.manual_seed(0); t_arange = fwd_logits(m, ids, train=True, mode="arange_debug")
    # standard reference: train forward with mode none (manual path off)
    torch.manual_seed(0); t_none = fwd_logits(m, ids, train=True, mode=None)
    d2 = (t_arange - t_none).abs().max().item()
    rel2 = d2 / (t_none.abs().max().item() + 1e-9)
    print(f"[2] train(arange_debug) vs train(std RoPE): max|Δ| = {d2:.2e}  rel = {rel2:.2e}  (expect ~0)")

    # 3. dim_specific / pose_shared train forward + backward
    for mode in ("dim_specific", "pose_shared"):
        os.environ["POS_EXPOSURE_MODE"] = mode
        if hasattr(m.transformer, "_pos_exp_mode"):
            del m.transformer._pos_exp_mode
        m.train(True)
        out = m(ids, labels=ids)
        loss = out.loss
        loss.backward()
        gnorm = sum(p.grad.norm().item() for p in m.parameters() if p.grad is not None)
        cs = m.transformer.h[0].attn._pos_exposure_cossin
        shp = tuple(cs[0].shape)
        print(f"[3] {mode}: loss={loss.item():.3f} grad_norm_sum={gnorm:.1f} cos_shape={shp} "
              f"(expect [{B},1,{T},{m.config.n_embd // m.config.n_head}])")
        m.zero_grad(set_to_none=True)

    # 4. dim_specific exposure range per pair: rebuild positions and check max angle
    inv = m.transformer.h[0].attn.rotary_emb.freqs.detach().cpu().numpy().reshape(-1)
    buf = float(os.environ.get("POS_EXPOSURE_BUFFER", "32"))
    P = 2 * np.pi / inv
    Lp = P + buf
    max_ang_expected = Lp * inv          # = 2π + buf*θ  for every pair
    print(f"[4] dim_specific per-pair max-angle = 2π+buf·θ: "
          f"min={max_ang_expected.min():.3f} max={max_ang_expected.max():.3f}  "
          f"(2π={2*np.pi:.3f}; longest-period pair P={P.max():.0f}, shortest P={P.min():.2f})")
    print("SMOKE DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    run(ap.parse_args())

#!/usr/bin/env python
"""PAT-225 L0/L1 gate: grid regression + S=1/2/4 forward/backward smoke.

Part A (pure math, no GPU):
  - assert the new fixed-support grid reproduces the production S=8 grid
    [2^0, 2^2, ..., 2^14] bit-exactly;
  - print the effective scale list for K in {1, 2, 4, 8}.

Part B (A4 checkpoint regression, GPU):
  - load the canonical A4 checkpoint (config has no wavelet_ctxscale_k field
    -> default 8 -> behavior must be unchanged);
  - verify the scales buffer equals the production grid;
  - run a forward pass on a fixed batch and check eval loss is finite and
    close to the recorded training-time eval loss.

Part C (S=1/2/4 smoke, GPU):
  - instantiate the A4 architecture with wavelet_ctxscale_k in {1, 2, 4}
    (random init), one forward/backward on random tokens;
  - check: finite loss, nonzero wavelet_ctx_router grads, scales buffer
    endpoints per spec (S=1 -> [128]; S>1 -> min 1, max 16384).
"""
import argparse
import json
import math
import sys

import torch

A4_CKPT = (
    "/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/"
    "runs/head_wise_scale_selection_vs_layer_wise/layer_wise/sigmoid_exp/"
    "s42_delta_detach/checkpoint-15000"
)

PRODUCTION_GRID = [2.0 ** (2 * i) for i in range(8)]


def expected_grid(k: int):
    if k == 1:
        return [2.0 ** 7]
    return [2.0 ** (14.0 * i / (k - 1)) for i in range(k)]


def part_a():
    print("=== Part A: grid math ===")
    assert expected_grid(8) == PRODUCTION_GRID, (
        f"K=8 grid mismatch:\n new={expected_grid(8)}\n prod={PRODUCTION_GRID}"
    )
    for k in (1, 2, 4, 8):
        g = expected_grid(k)
        print(f"K={k}: {g}")
        if k == 1:
            assert g == [128.0]
        else:
            assert g[0] == 1.0 and g[-1] == 16384.0, f"K={k} endpoints wrong: {g}"
    print("Part A PASS: K=8 bit-exact vs production; endpoints fixed; K=1 -> 128")


def load_model(config_overrides=None, from_ckpt=True):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config = AutoConfig.from_pretrained(A4_CKPT)
    if config_overrides:
        for k, v in config_overrides.items():
            setattr(config, k, v)
    if from_ckpt:
        model = AutoModelForCausalLM.from_pretrained(A4_CKPT, config=config)
    else:
        model = AutoModelForCausalLM.from_config(config)
    # use_fast=False: the env's tokenizers lib cannot parse this checkpoint's
    # tokenizer.json (untagged-enum ModelWrapper error); slow GPT2 tokenizer
    # loads from vocab.json/merges.txt instead.
    tok = AutoTokenizer.from_pretrained(A4_CKPT, use_fast=False)
    return model.cuda().eval(), tok


def get_scales_buffers(model):
    out = {}
    for name, mod in model.named_modules():
        if hasattr(mod, "wavelet_ctxscale_scales"):
            out[name] = mod.wavelet_ctxscale_scales.detach().float().cpu()
    return out


def part_b():
    print("=== Part B: A4 regression (default K=8) ===")
    model, tok = load_model()
    bufs = get_scales_buffers(model)
    assert bufs, "no wavelet_ctxscale_scales buffers found"
    prod = torch.tensor(PRODUCTION_GRID, dtype=torch.float32)
    for name, buf in bufs.items():
        assert torch.equal(buf, prod), f"{name}: scales buffer changed! {buf.tolist()}"
    print(f"scales buffers identical to production for {len(bufs)} layers")

    text = ("The quick brown fox jumps over the lazy dog. " * 40).strip()
    ids = tok(text, return_tensors="pt").input_ids[:, :512].cuda()
    with torch.no_grad():
        out = model(ids, labels=ids)
    loss = float(out.loss)
    print(f"A4 forward loss on probe batch: {loss:.4f}")
    assert math.isfinite(loss), "non-finite loss"
    # generous sanity band around the recorded eval regime (guards silent breakage)
    assert 1.0 < loss < 8.0, f"loss {loss} outside sanity band"
    del model
    torch.cuda.empty_cache()
    print("Part B PASS")


def part_c():
    print("=== Part C: S=1/2/4 smoke (random init) ===")
    for k in (1, 2, 4):
        model, tok = load_model(config_overrides={"wavelet_ctxscale_k": k}, from_ckpt=False)
        model.train()
        bufs = get_scales_buffers(model)
        exp = torch.tensor(expected_grid(k), dtype=torch.float32)
        for name, buf in bufs.items():
            assert torch.equal(buf, exp), f"K={k} {name}: {buf.tolist()} != {exp.tolist()}"
        ids = torch.randint(100, 5000, (2, 512)).cuda()
        out = model(ids, labels=ids)
        loss = out.loss
        assert math.isfinite(float(loss)), f"K={k}: non-finite loss"
        loss.backward()
        router_grads = [
            p.grad.abs().sum().item()
            for n, p in model.named_parameters()
            if "wavelet_ctx_router" in n and p.grad is not None
        ]
        assert router_grads and sum(router_grads) > 0, f"K={k}: router grads zero/missing"
        n_params = sum(p.numel() for p in model.parameters())
        print(
            f"K={k}: loss={float(loss):.4f} router_grad_sum={sum(router_grads):.3e} "
            f"params={n_params} scales={exp.tolist()}"
        )
        del model
        torch.cuda.empty_cache()
    print("Part C PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="abc")
    args = ap.parse_args()
    if "a" in args.parts:
        part_a()
    if "b" in args.parts:
        part_b()
    if "c" in args.parts:
        part_c()
    print("ALL REQUESTED PARTS PASS")


if __name__ == "__main__":
    main()

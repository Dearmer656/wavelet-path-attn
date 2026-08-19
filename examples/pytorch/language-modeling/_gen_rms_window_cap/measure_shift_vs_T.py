#!/usr/bin/env python3
"""Measure how the wavelet shift (beta_m) behaves at L_train vs longer L_test.

Hypothesis (from reading path_attn.py's default 'legacy' wavelet_shift_T_mode):
    beta_upper = T_test - 1   (NOT capped to L_train)
    beta_m = round(rho * beta_upper), rho in [0,1] learned per-query
So if rho's learned distribution stays roughly fixed, beta_m should scale
*linearly* with T_test - i.e. shifts at L4096 should be ~8x those at L512,
purely mechanically, not because the model "learned" to shift further.

This script forwards the same checkpoint at T=512, 2048, 4096 on real
wikitext-derived text, dumps rho and beta_m per layer, and reports:
  - rho distribution (should be ~invariant across T if hypothesis is right)
  - beta_m distribution (should scale ~linearly with T-1)
  - beta_m / (T-1) ratio (should be ~constant == rho, confirming the formula)
"""
import sys
import numpy as np
import torch

sys.path.insert(0, "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling")
import run_clm  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

CHECKPOINT = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat244_dual_temp/K3_L512_fixedratioRmsBoth_128_256_384_s42/checkpoint-15000"
CFG_PATH = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat244_dual_temp/K3_L512_fixedratioRmsBoth_128_256_384_s42/supply_model.cfg"
N_LAYERS = 12
LENGTHS = [512, 2048, 4096]
SCALES = [128.0, 256.0, 384.0]  # fixedratioRmsBoth's production scale bank


def load_model(checkpoint_dir, cfg_path, block_size):
    config = AutoConfig.from_pretrained(checkpoint_dir)
    cfg = run_clm.read_kv_config(str(cfg_path))
    run_clm.add_missing_to_hf_config(config, cfg)
    config.attn_implementation = "path_attn"
    config.use_cache = False
    config.block_size = block_size
    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, config=config)
    return model


def build_batch(tokenizer, length, n=2):
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    text_parts = [row["text"] for row in dataset if row.get("text", "").strip()]
    corpus = "\n\n".join(text_parts)
    token_ids = tokenizer(corpus, add_special_tokens=False)["input_ids"]
    needed = n * length
    token_ids = token_ids[:needed]
    batch = torch.tensor(token_ids, dtype=torch.long).view(n, length)
    return batch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"{'T':>6} {'layer':>5} {'rho_mean':>9} {'rho_std':>8} {'beta_mean':>10} {'beta_std':>9} {'beta/(T-1)':>11} {'beta_upper=T-1':>15}")
    results = {}
    for T in LENGTHS:
        model = load_model(CHECKPOINT, CFG_PATH, T)
        model.eval().to(device)
        batch = build_batch(tokenizer, T, n=4)
        with torch.no_grad():
            model(input_ids=batch.to(device))

        results[T] = {}
        for layer in range(N_LAYERS):
            core = getattr(model.transformer.h[layer].attn, "core", model.transformer.h[layer].attn)
            rho = getattr(core, "_last_ctxscale_rho", None)
            beta_m = getattr(core, "_last_ctxscale_beta_m", None)
            if rho is None or beta_m is None:
                print(f"{T:>6} {layer:>5}  MISSING HOOKS")
                continue
            rho_np = rho.detach().float().cpu().numpy().reshape(-1)
            beta_np = beta_m.detach().float().cpu().numpy().reshape(-1)
            ratio = beta_np / max(1, T - 1)
            abs_beta = np.abs(beta_np)
            pcts = np.percentile(abs_beta, [50, 90, 99, 100])
            print(
                f"{T:>6} {layer:>5} {rho_np.mean():>9.4f} {rho_np.std():>8.4f} "
                f"{beta_np.mean():>10.2f} {beta_np.std():>9.2f} {ratio.mean():>11.4f} {T-1:>15d}"
                f"  | |beta| p50={pcts[0]:.3f} p90={pcts[1]:.3f} p99={pcts[2]:.3f} max={pcts[3]:.3f}"
                f"  | max*scale(384)={pcts[3]*384:.1f} tok  p99*scale(384)={pcts[2]*384:.1f} tok"
            )
            results[T][layer] = {
                "rho_mean": float(rho_np.mean()),
                "rho_std": float(rho_np.std()),
                "beta_mean": float(beta_np.mean()),
                "beta_std": float(beta_np.std()),
                "ratio_mean": float(ratio.mean()),
                "abs_beta_p50": float(pcts[0]),
                "abs_beta_p90": float(pcts[1]),
                "abs_beta_p99": float(pcts[2]),
                "abs_beta_max": float(pcts[3]),
            }
        del model
        torch.cuda.empty_cache()

    print("\n=== Summary: mean beta_m across layers, per T, and scaling factor vs L_train=512 ===")
    base_T = LENGTHS[0]
    base_betas = np.mean([results[base_T][l]["beta_mean"] for l in results[base_T]])
    for T in LENGTHS:
        betas = np.mean([results[T][l]["beta_mean"] for l in results[T]])
        rhos = np.mean([results[T][l]["rho_mean"] for l in results[T]])
        print(f"T={T:>5}: mean beta_m across layers = {betas:>9.2f}  (x{betas/max(base_betas,1e-9):.2f} vs T={base_T})  |  mean rho = {rhos:.4f}  |  T-1={T-1}")

    print("\n=== Tail check: worst-case |beta_m| across all layers, per T (and implied token shift per scale) ===")
    for T in LENGTHS:
        max_p99 = max(results[T][l]["abs_beta_p99"] for l in results[T])
        max_max = max(results[T][l]["abs_beta_max"] for l in results[T])
        worst_layer_p99 = max(results[T], key=lambda l: results[T][l]["abs_beta_p99"])
        worst_layer_max = max(results[T], key=lambda l: results[T][l]["abs_beta_max"])
        print(f"T={T:>5}: max-over-layers |beta| p99={max_p99:.3f} (layer {worst_layer_p99})  |  true max={max_max:.3f} (layer {worst_layer_max})")
        for s in SCALES:
            print(f"          -> at scale={s:.0f}: p99 token-shift={max_p99*s:>7.1f}  max token-shift={max_max*s:>7.1f}")


if __name__ == "__main__":
    main()

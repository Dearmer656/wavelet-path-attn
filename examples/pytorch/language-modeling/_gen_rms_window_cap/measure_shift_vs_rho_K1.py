#!/usr/bin/env python3
"""Measure how the wavelet shift (beta_m) behaves across different K1 rho values,
at both L_train (512) and longer L_test (2048, 4096).

Companion to measure_shift_vs_T.py (which fixed rho=[128,256,384] combined and
varied T). This script fixes T and varies rho (single-scale K1 checkpoints),
to see whether the *scale* of the wavelet itself affects the shift distribution
(rho, beta_m) independently of the T-invariance already established.
"""
import sys
import numpy as np
import torch

sys.path.insert(0, "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling")
import run_clm  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

CHECKPOINTS = {
    128: "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat244_dual_temp/K1_L512_me14_rho128_ricker_s42/checkpoint-15000",
    256: "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat244_dual_temp/K1_L512_me16_rho256_ricker_s42/checkpoint-15000",
    384: "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat244_dual_temp/K1_L512_me17p1699_rho384_ricker_s42/checkpoint-15000",
}
CFGS = {
    128: "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat244_dual_temp/K1_L512_me14_rho128_ricker_s42/supply_model.cfg",
    256: "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat244_dual_temp/K1_L512_me16_rho256_ricker_s42/supply_model.cfg",
    384: "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat244_dual_temp/K1_L512_me17p1699_rho384_ricker_s42/supply_model.cfg",
}
N_LAYERS = 12
LENGTHS = [512, 2048, 4096]


def load_model(checkpoint_dir, cfg_path, block_size):
    config = AutoConfig.from_pretrained(checkpoint_dir)
    cfg = run_clm.read_kv_config(str(cfg_path))
    run_clm.add_missing_to_hf_config(config, cfg)
    config.attn_implementation = "path_attn"
    config.use_cache = False
    config.block_size = block_size
    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, config=config)
    return model


def build_batch(tokenizer, length, n=4):
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

    print(f"{'rho':>5} {'T':>6} {'layer':>5} {'rho_val_mean':>12} {'beta_mean':>10} {'beta_std':>9} {'|beta|p50':>10} {'|beta|p90':>10} {'|beta|p99':>10} {'|beta|max':>10} {'tok_p99':>9} {'tok_max':>9}")
    all_results = {}
    for rho_scale, checkpoint in CHECKPOINTS.items():
        cfg_path = CFGS[rho_scale]
        all_results[rho_scale] = {}
        for T in LENGTHS:
            model = load_model(checkpoint, cfg_path, T)
            model.eval().to(device)
            batch = build_batch(tokenizer, T, n=4)
            with torch.no_grad():
                model(input_ids=batch.to(device))

            all_results[rho_scale][T] = {}
            for layer in range(N_LAYERS):
                core = getattr(model.transformer.h[layer].attn, "core", model.transformer.h[layer].attn)
                rho_t = getattr(core, "_last_ctxscale_rho", None)
                beta_m = getattr(core, "_last_ctxscale_beta_m", None)
                if rho_t is None or beta_m is None:
                    print(f"{rho_scale:>5} {T:>6} {layer:>5}  MISSING HOOKS")
                    continue
                rho_np = rho_t.detach().float().cpu().numpy().reshape(-1)
                beta_np = beta_m.detach().float().cpu().numpy().reshape(-1)
                abs_beta = np.abs(beta_np)
                pcts = np.percentile(abs_beta, [50, 90, 99, 100])
                tok_p99 = pcts[2] * rho_scale
                tok_max = pcts[3] * rho_scale
                print(
                    f"{rho_scale:>5} {T:>6} {layer:>5} {rho_np.mean():>12.4f} "
                    f"{beta_np.mean():>10.3f} {beta_np.std():>9.3f} "
                    f"{pcts[0]:>10.3f} {pcts[1]:>10.3f} {pcts[2]:>10.3f} {pcts[3]:>10.3f} "
                    f"{tok_p99:>9.1f} {tok_max:>9.1f}"
                )
                all_results[rho_scale][T][layer] = {
                    "rho_mean": float(rho_np.mean()),
                    "beta_mean": float(beta_np.mean()),
                    "beta_std": float(beta_np.std()),
                    "abs_beta_p50": float(pcts[0]),
                    "abs_beta_p90": float(pcts[1]),
                    "abs_beta_p99": float(pcts[2]),
                    "abs_beta_max": float(pcts[3]),
                }
            del model
            torch.cuda.empty_cache()

    print("\n=== Summary: mean |beta| (p90) across layers, per rho and T ===")
    for rho_scale in CHECKPOINTS:
        for T in LENGTHS:
            d = all_results[rho_scale][T]
            mean_p90 = np.mean([d[l]["abs_beta_p90"] for l in d])
            mean_p99 = np.mean([d[l]["abs_beta_p99"] for l in d])
            max_p99_layer = max(d, key=lambda l: d[l]["abs_beta_p99"])
            print(
                f"rho={rho_scale:>3} T={T:>5}: mean-over-layers |beta| p90={mean_p90:.3f} p99={mean_p99:.3f} "
                f"| worst layer={max_p99_layer} p99={d[max_p99_layer]['abs_beta_p99']:.3f} "
                f"-> tok_p99_worst={d[max_p99_layer]['abs_beta_p99']*rho_scale:.1f}"
            )


if __name__ == "__main__":
    main()

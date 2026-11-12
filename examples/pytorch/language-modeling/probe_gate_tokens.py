#!/usr/bin/env python3
"""PAT-310: per-layer, per-token-position QWAB gate for the three wavelet-center placements.

Same model loading, prompt construction (front-pinned eval prompt, first n examples per length) and gate hook
(_pat_g0_cap) as probe_gate_by_offset.py, but instead of three summaries it keeps, per example and layer:
  * g_all : mean gate over all prompt tokens
  * g_bins: mean gate in absolute token-position bins of width BIN (aligned across lengths)
  * g_ans : gate at the answer-generating position (last prompt token)
Placements: first position (QWAB paper seeds), i/2 (center05), query (center10), seeds 42-44.
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch

import probe_gate_by_offset as pgo
from probe_gate_by_offset import rc, pa, AutoConfig, AutoModelForCausalLM, AutoTokenizer, MODELS, apply_supply_cfg, HOTPOT

R = Path("/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat310_center")
for s in (42, 43, 44):
    MODELS[f"C05_s{s}"] = R / f"K1_L512_me16_rho256_ricker_center05_s{s}"
    MODELS[f"C10_s{s}"] = R / f"K1_L512_me16_rho256_ricker_center10_s{s}"
BIN = 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(f"{p}_s{s}" for p in ("QWAB", "C05", "C10") for s in (42, 43, 44)))
    ap.add_argument("--lens", default="512,4096")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="/cl/work5/hongyu-s/tmp_pat310/gate_tokens")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    rc.GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained("gpt2"); tok = rc._get_tokenizer()
    rows = {}
    with open(HOTPOT) as f:
        for line in f:
            ex = json.loads(line); rows.setdefault(ex["meta"]["target_total_tokens"], []).append(ex)
    for name in a.models.split(","):
        ck = MODELS[name] / "checkpoint-15000"
        cfg = AutoConfig.from_pretrained(ck); apply_supply_cfg(cfg, ck.parent)
        cfg.attn_implementation = "path_attn"; cfg.path_attn_impl = "pytorch"
        cfg.wavelet_logit_bias_log_every = 10 ** 9; cfg.wavelet_ctxscale_eval_log_once = False
        model = AutoModelForCausalLM.from_pretrained(ck, config=cfg, torch_dtype=torch.float32).to(dev).eval()
        layers = [m for m in model.modules() if isinstance(m, pa.PaTHAttention)]
        for m in layers: m._capture_debug_tensors = False
        for L in map(int, a.lens.split(",")):
            f = out / f"{name}_L{L}.json"
            if f.exists(): print(f"[have] {f.name}"); continue
            recs = []
            for ex in rows[L][:a.n]:
                ids, c0, c1, sup0 = pgo.build_prompt(ex, L, tok, (0, 0))
                for m in layers: m._pat_g0_cap = {}
                with torch.no_grad(): model(input_ids=ids.to(dev))
                g = np.stack([m._pat_g0_cap["g0_gate"][-1][1][0, :, 0].numpy() for m in layers])  # [layers, T]
                for m in layers: m._pat_g0_cap = None
                T = g.shape[1]; nb = -(-T // BIN)
                bins = [g[:, b * BIN:min((b + 1) * BIN, T)].mean(1).tolist() for b in range(nb)]  # [nb][layers]
                recs.append(dict(T=int(T), c0=int(c0), c1=int(c1), g_all=g.mean(1).tolist(), g_ans=g[:, -1].tolist(),
                                 g_bins=bins))
            f.write_text(json.dumps({"model": name, "L": L, "bin": BIN, "recs": recs}))
            ga = np.array([r["g_all"] for r in recs])
            print(f"[done] {f.name} n={len(recs)} g_all mean={ga.mean():.4f} per-layer={np.round(ga.mean(0), 3).tolist()}", flush=True)
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

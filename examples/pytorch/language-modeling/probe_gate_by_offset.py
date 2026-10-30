#!/usr/bin/env python3
"""PAT-310 (reviewer 1Pzy): inference-time QWAB gate statistics by evidence position.

Builds the HotpotQA-Long generation prompt exactly as the eval does
("Context:\n" + build_context_budgeted(...) + "\nQuestion: q\nAnswer:"), front-pinned
(offset 0) or with the support block shifted by support_offset_range, and records the
per-layer query-conditioned gate g0(q) (head-shared, [B,T,1]) via the _pat_g0_cap hook.

Per example: gate at the answer-generating position (last prompt token), mean gate over
question tokens, mean gate over context tokens, and the token index where the support
block starts.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

sys.argv0 = list(sys.argv); sys.argv = ["x"]
sys.path.insert(0, "/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling")
sys.path.insert(0, "/project/nlp-work5/hongyu-s/transformers/src")
sys.path.insert(0, "/project/nlp-work5/hongyu-s/flash-linear-attention")
import run_clm as rc
import fla.layers.path_attn as pa
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from probe_path_transport_far_near import MODELS, apply_supply_cfg
sys.argv = sys.argv0

HOTPOT = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/hotpot_long_dev.jsonl"


def build_prompt(ex, L, tok, rng):
    pre = tok("Context:\n", add_special_tokens=True)["input_ids"]
    q_ids = tok(f"\nQuestion: {ex['question']}\n", add_special_tokens=False)["input_ids"]
    a_ids = tok("Answer:", add_special_tokens=False)["input_ids"]
    budget = L - len(pre) - len(q_ids) - len(a_ids) - 16
    txt, cids, st = rc.build_context_budgeted(ex, budget, prefer_same_title=True, min_tokens=budget - 64,
                                              support_offset_range=rng)
    sf = rc._get_supporting_pairs(ex); t0 = sf[0][0]; s0 = rc._build_title2sents(ex)[t0][sf[0][1]]
    p = txt.find(f"{t0}: {s0}")
    sup0 = len(pre) + (len(tok(txt[:p], add_special_tokens=False)["input_ids"]) if p >= 0 else -10 ** 6)
    ids = pre + list(cids) + q_ids + a_ids
    return torch.tensor([ids]), len(pre), len(pre) + len(cids), sup0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="QWAB_s42,QWAB_s43,QWAB_s44")
    ap.add_argument("--conds", default="512:0:0,2048:0:0,4096:0:0,4096:0:256,4096:256:512,4096:512:1024")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="/cl/work5/hongyu-s/tmp_pat310/gate_probe")
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
        for c in a.conds.split(","):
            L, lo, hi = map(int, c.split(":"))
            f = out / f"{name}_L{L}_off{lo}_{hi}.json"
            if f.exists(): print(f"[have] {f.name}"); continue
            recs = []
            for ex in rows[L][:a.n]:
                ids, c0, c1, sup0 = build_prompt(ex, L, tok, (lo, hi))
                for m in layers: m._pat_g0_cap = {}
                with torch.no_grad(): model(input_ids=ids.to(dev))
                g = np.stack([m._pat_g0_cap["g0_gate"][-1][1][0, :, 0].numpy() for m in layers])  # [layers, T]
                for m in layers: m._pat_g0_cap = None
                recs.append(dict(T=int(ids.shape[1]), sup0=int(sup0), g_ans=g[:, -1].tolist(),
                                 g_q=g[:, c1:].mean(1).tolist(), g_ctx=g[:, c0:c1].mean(1).tolist()))
            f.write_text(json.dumps({"model": name, "L": L, "off": [lo, hi], "recs": recs}))
            ga = np.array([r["g_ans"] for r in recs])
            print(f"[done] {f.name} n={len(recs)} g_ans mean={ga.mean():.4f} per-layer={np.round(ga.mean(0), 3).tolist()}", flush=True)
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

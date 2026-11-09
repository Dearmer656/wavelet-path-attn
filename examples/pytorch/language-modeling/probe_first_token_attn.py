#!/usr/bin/env python3
"""PAT-310 (reviewer 1Pzy RC4 / XHxZ attention sink): first-token attention share, PaTH vs QWAB, GPT-2 small.

Metric follows Qiu et al. (NeurIPS 2025, Table 4 "F-Attn"): attention probability assigned to the first token,
averaged over query positions (query 0 excluded, it can only attend to itself), heads and layers.
Also recorded: the same share at the answer-generating position (last prompt token), and the share of the first
64 tokens (prefix), per layer.

Probabilities are the softmaxed logits actually used by the PaTH attention output in the pytorch implementation
(P_wav: Householder transport + QWAB bias for QWAB, transport only for PaTH), read via
PaTHAttention._maybe_capture_entmax_attention_stats. The prompt is the front-pinned HotpotQA-Long eval prompt,
built exactly as in probe_gate_by_offset.py (offset 0).
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
PREFIX = 64


def build_prompt(ex, L, tok):
    pre = tok("Context:\n", add_special_tokens=True)["input_ids"]
    q_ids = tok(f"\nQuestion: {ex['question']}\n", add_special_tokens=False)["input_ids"]
    a_ids = tok("Answer:", add_special_tokens=False)["input_ids"]
    budget = L - len(pre) - len(q_ids) - len(a_ids) - 16
    txt, cids, st = rc.build_context_budgeted(ex, budget, prefer_same_title=True, min_tokens=budget - 64,
                                              support_offset_range=(0, 0))
    return torch.tensor([pre + list(cids) + q_ids + a_ids])


CAP = {}


def cap(self, probs, future, layer_idx):
    lid = int(self.layer_idx if layer_idx is None else layer_idx)
    assert probs.dim() == 4 and probs.shape[0] == 1 and probs.shape[-2] == probs.shape[-1], tuple(probs.shape)
    p = probs[0].detach().float()                      # [H, T, T]
    T = p.shape[-1]
    first = p[:, 1:, 0]                                # [H, T-1]  (query 0 excluded)
    CAP[lid] = dict(first_all=first.mean().item(),
                    first_ans=p[:, T - 1, 0].mean().item(),
                    prefix_all=p[:, 1:, :PREFIX].sum(-1).mean().item(),
                    prefix_ans=p[:, T - 1, :PREFIX].sum(-1).mean().item(),
                    first_all_head=first.mean(1).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="PaTH_s42,PaTH_s43,PaTH_s44,QWAB_s42,QWAB_s43,QWAB_s44")
    ap.add_argument("--lens", default="512,2048,4096")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="/cl/work5/hongyu-s/tmp_pat310/first_token_attn")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    rc.GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained("gpt2"); tok = rc._get_tokenizer()
    rows = {}
    with open(HOTPOT) as f:
        for line in f:
            ex = json.loads(line); rows.setdefault(ex["meta"]["target_total_tokens"], []).append(ex)
    pa.PaTHAttention._maybe_capture_entmax_attention_stats = cap
    for name in a.models.split(","):
        ck = MODELS[name] / "checkpoint-15000"
        cfg = AutoConfig.from_pretrained(ck); apply_supply_cfg(cfg, ck.parent)
        if not hasattr(cfg, "wavelet_ctxscale_k"):  # PaTH baseline: ckpt carries unused K=8 ctxscale params (same as probe_path_transport_far_near.run_model)
            cfg.wavelet_ctxscale_k = 8
            mx = getattr(cfg, "wavelet_ctxscale_scale_max_exp", 14.0)
            if not isinstance(mx, (list, tuple)): cfg.wavelet_ctxscale_scale_max_exp = [float(mx)] * 8
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
                ids = build_prompt(ex, L, tok)
                CAP.clear()
                with torch.no_grad(): model(input_ids=ids.to(dev))
                assert sorted(CAP) == list(range(len(layers))), (name, L, sorted(CAP))
                recs.append(dict(T=int(ids.shape[1]), layers=[CAP[l] for l in range(len(layers))]))
            f.write_text(json.dumps({"model": name, "L": L, "recs": recs}))
            fa = np.array([[l["first_all"] for l in r["layers"]] for r in recs])
            print(f"[done] {f.name} n={len(recs)} first-token share (all queries) mean={fa.mean():.4f} per-layer={np.round(fa.mean(0), 3).tolist()}", flush=True)
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

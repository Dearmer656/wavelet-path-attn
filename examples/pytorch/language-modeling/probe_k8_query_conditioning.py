#!/usr/bin/env python3
"""
Does the K=8 router actually QUERY-CONDITION its scale choice, or is the flat
mean just a fixed per-layer preference?

The mean-over-tokens heatmap cannot tell these apart. Here we measure the
CROSS-TOKEN DISPERSION of the per-token scale distribution:
  per layer: w_t = softmax-normalized pi_scale for token t (dist over K scales)
             w_bar = mean_t w_t
             DISP = mean_t KL(w_t || w_bar)      <- query-conditioning strength
             also: mean per-token top1 (peakedness) and argmax-scale entropy.
Large DISP => tokens pick genuinely different scales (query-conditioning real).
DISP ~ 0   => every token picks the same distribution (fixed preference).

Usage: probe_k8_query_conditioning.py {hotpot|xsum} [max_ctx_tokens]
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

CKPT = "/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/head_wise_scale_selection_vs_layer_wise/layer_wise/sigmoid_exp/s42_delta_detach/checkpoint-15000"
HOTPOT = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/hotpot_long_dev_uniform.jsonl"
XSUM = "/cl/work5/hongyu-s/fact-check-summarization/xsum_test_filter_level2_official_style.jsonl"

DATASET = sys.argv[1] if len(sys.argv) > 1 else "hotpot"
MAX_CTX = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
N_SAMPLES = 8
EPS = 1e-8


def hotpot_docs(tok):
    out = []
    with open(HOTPOT) as f:
        for line in f:
            r = json.loads(line)
            if r.get("meta", {}).get("target_total_tokens") != 2048: continue
            ctx = "".join(f"Title: {t}\n{' '.join(s).strip()}\n\n" for t, s in r["context"])
            out.append(f"Context:\n{ctx}\nQuestion: {r['question']}\nAnswer: {r.get('answer','')}")
            if len(out) >= N_SAMPLES: break
    return out


def xsum_docs(tok):
    out = []
    with open(XSUM) as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            doc = r.get("document", ""); summ = (r.get("summary_original") or "").strip()
            if not doc or not summ: continue
            doc = tok.decode(tok(doc)["input_ids"][:MAX_CTX])
            out.append(f"Summarize the following document:\n{doc}\n\nSummary: {summ}")
            if len(out) >= N_SAMPLES: break
    return out


def main():
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(CKPT); cfg.attn_implementation = "path_attn"
    cfg.wavelet_logit_bias_log_every = 10 ** 9; cfg.wavelet_ctxscale_eval_log_once = False
    model = AutoModelForCausalLM.from_pretrained(CKPT, config=cfg, torch_dtype=torch.float32).to(dev).eval()
    layers, scales = {}, None
    for m in model.modules():
        if isinstance(m, pa.PaTHAttention):
            layers[int(getattr(m, "layer_idx", -1) or -1)] = m
            if scales is None and hasattr(m, "wavelet_ctxscale_scales"):
                scales = m.wavelet_ctxscale_scales.detach().float().cpu().numpy()
    L = max(layers) + 1; K = len(scales)
    docs = hotpot_docs(tok) if DATASET == "hotpot" else xsum_docs(tok)
    print(f"dataset={DATASET} cases={len(docs)} L={L} K={K}")

    # accumulate per-layer stacked per-token distributions
    W = {l: [] for l in range(L)}
    for d in docs:
        ids = tok(d, return_tensors="pt", truncation=False)["input_ids"][:, :MAX_CTX + 128].to(dev)
        with torch.no_grad(): model(input_ids=ids)
        for lid, m in layers.items():
            rp = getattr(m, "_last_ctxscale_router_prob", None)
            if rp is None: continue
            r = rp[0].detach().float()                 # [T,H,K] or [T,K]
            if r.dim() == 3: r = r.mean(dim=1)          # mean over heads -> [T,K]
            r = r[8:]                                   # drop first few degenerate positions
            s = r.sum(dim=-1, keepdim=True)
            keep = (s.squeeze(-1) > 1e-4)               # drop near-null tokens (no scale mass)
            r = r[keep]; s = s[keep]
            if r.shape[0] == 0: continue
            w = (r / s.clamp_min(EPS)).cpu()            # normalized dist over scales [n,K]
            W[lid].append(w)

    print("\n=== query-conditioning of scale choice (cross-token dispersion) ===")
    print("layer | DISP=mean_t KL(w_t||w_bar) | top1_mean | argmax_entropy(bits, max=%.2f) | w_bar(top scale)" % np.log2(K))
    for lid in range(L):
        if not W[lid]:
            print(f"  L{lid:>2} | (no active tokens)"); continue
        w = torch.cat(W[lid], dim=0).numpy()           # [N,K]
        w_bar = w.mean(0)                               # [K]
        # per-token KL(w_t || w_bar)
        kl = np.sum(w * (np.log(w + EPS) - np.log(w_bar + EPS)), axis=1)
        disp = float(np.mean(kl))
        top1 = float(np.mean(w.max(axis=1)))            # per-token peakedness
        am = w.argmax(axis=1)
        hist = np.bincount(am, minlength=K).astype(float); hist /= hist.sum()
        am_ent = float(-np.sum(hist * np.log2(hist + EPS)))  # spread of top-choice across tokens
        top_scale = int(scales[int(np.argmax(w_bar))])
        print(f"  L{lid:>2} | {disp:20.4f} | {top1:9.3f} | {am_ent:28.3f} | ρ{top_scale}")
    print("\n>> DISP large & argmax_entropy high => query-conditioning REAL (tokens pick different scales)")
    print(">> DISP ~0 & argmax_entropy ~0     => fixed per-layer preference (query-conditioning redundant)")


if __name__ == "__main__":
    main()

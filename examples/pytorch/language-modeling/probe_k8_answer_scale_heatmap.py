#!/usr/bin/env python3
"""
K=8 scale selection AT THE ANSWER/TARGET POSITIONS (not full-length).

Hypothesis (user): context processing is homogeneous; differentiation shows up
only where the model produces the answer/summary. So teacher-force prompt+target
and read _last_ctxscale_router_prob only at the target-token positions.

Datasets:
  hotpot: "Context:\\n{ctx}\\nQuestion: {q}\\nAnswer:" + " {answer}"
  xsum  : "Summarize the following document:\\n{doc}\\n\\nSummary:" + " {summary}"

Usage: probe_k8_answer_scale_heatmap.py {hotpot|xsum} [max_ctx_tokens]
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
OUT = Path("/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat234_scale_card")

DATASET = sys.argv[1] if len(sys.argv) > 1 else "hotpot"
MAX_CTX = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
N_SAMPLES = 16


def hotpot_pairs(tok):
    out = []
    with open(HOTPOT) as f:
        for line in f:
            r = json.loads(line)
            if r.get("meta", {}).get("target_total_tokens") != 2048:
                continue
            ans = str(r.get("answer", "")).strip()
            if not ans:
                continue
            ctx = "".join(f"Title: {t}\n{' '.join(s).strip()}\n\n" for t, s in r["context"])
            prompt = f"Context:\n{ctx}\nQuestion: {r['question']}\nAnswer:"
            out.append((prompt, " " + ans))
            if len(out) >= N_SAMPLES:
                break
    return out


def xsum_pairs(tok):
    out = []
    with open(XSUM) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            doc = r.get("document", "")
            summ = (r.get("summary_original") or r.get("summary_filtered") or "").strip()
            if not doc or not summ:
                continue
            # cap the document so prompt+summary stays reasonable
            doc_ids = tok(doc)["input_ids"][:MAX_CTX]
            doc = tok.decode(doc_ids)
            prompt = f"Summarize the following document:\n{doc}\n\nSummary:"
            out.append((prompt, " " + summ))
            if len(out) >= N_SAMPLES:
                break
    return out


def main():
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    cfg = AutoConfig.from_pretrained(CKPT)
    cfg.attn_implementation = "path_attn"
    cfg.wavelet_logit_bias_log_every = 10 ** 9
    cfg.wavelet_ctxscale_eval_log_once = False
    model = AutoModelForCausalLM.from_pretrained(CKPT, config=cfg, torch_dtype=torch.float32).to(dev).eval()

    layers, scales = {}, None
    for m in model.modules():
        if isinstance(m, pa.PaTHAttention):
            _lid = getattr(m, "layer_idx", None)
            if _lid is None: continue
            lid = int(_lid)
            layers[lid] = m
            if scales is None and hasattr(m, "wavelet_ctxscale_scales"):
                scales = m.wavelet_ctxscale_scales.detach().float().cpu().numpy()
    L = max(layers) + 1; K = len(scales)

    pairs = hotpot_pairs(tok) if DATASET == "hotpot" else xsum_pairs(tok)
    print(f"dataset={DATASET} cases={len(pairs)} L={L} K={K} scales={scales.astype(int).tolist()}")

    acc = np.zeros((L, K)); nacc = np.zeros((L, K)); null_acc = np.zeros(L); null_n = np.zeros(L)
    ntok_total = 0
    for prompt, target in pairs:
        p_ids = tok(prompt, return_tensors="pt", truncation=False)["input_ids"]
        f_ids = tok(prompt + target, return_tensors="pt", truncation=False)["input_ids"]
        a_start = p_ids.shape[1] - 1          # include the "Answer:"/"Summary:" onset token
        a_end = f_ids.shape[1]
        if a_end <= a_start + 1:               # empty target after tokenization
            continue
        ids = f_ids.to(dev)
        with torch.no_grad():
            model(input_ids=ids)
        ntok_total += (a_end - a_start)
        for lid, m in layers.items():
            rp = getattr(m, "_last_ctxscale_router_prob", None)
            if rp is None: continue
            r = rp[0].detach().float()         # [T,H,K] or [T,K]
            if r.dim() == 2: r = r.unsqueeze(1)
            r = r[a_start:a_end]               # target positions only -> [n_ans,H,K]
            acc[lid] += r.mean(dim=(0, 1)).cpu().numpy(); nacc[lid] += 1
            nm = getattr(m, "_last_ctxscale_null_mass", None)
            if nm is not None:
                nn = nm[0].detach().float()
                if nn.dim() >= 2: nn = nn.mean(dim=tuple(range(1, nn.dim())))
                null_acc[lid] += float(nn[a_start:a_end].mean().item()); null_n[lid] += 1

    mat = np.divide(acc, np.maximum(nacc, 1))
    null_mean = np.divide(null_acc, np.maximum(null_n, 1))
    print(f"avg target tokens/case = {ntok_total/max(len(pairs),1):.1f}")
    print("\n=== per-layer pi_scale at ANSWER positions ===")
    print("layer | " + " ".join(f"r{int(s):>6d}" for s in scales) + " | null | sum_g")
    for lid in range(L):
        print(f"  L{lid:>2} | " + " ".join(f"{mat[lid,j]:6.3f}" for j in range(K)) + f" | {null_mean[lid]:.3f} | {mat[lid].sum():.3f}")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 6))
        im = ax.imshow(mat, aspect="auto", cmap="viridis", origin="upper")
        ax.set_xticks(range(K)); ax.set_xticklabels([f"ρ{int(s)}" for s in scales], rotation=45, ha="right")
        ax.set_yticks(range(L)); ax.set_yticklabels([f"L{i}" for i in range(L)])
        ax.set_xlabel("scale ρ"); ax.set_ylabel("layer")
        ax.set_title(f"K=8 QWAB scale selection at ANSWER positions — {DATASET}")
        for i in range(L):
            for j in range(K):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        color="white" if mat[i, j] < mat.max() * 0.6 else "black", fontsize=7)
        fig.colorbar(im, ax=ax, label="π_scale (answer positions)")
        out = OUT / f"k8_answer_scale_heatmap_{DATASET}.png"
        fig.tight_layout(); fig.savefig(out, dpi=150)
        print(f"\nsaved: {out}")
    except Exception as e:
        print(f"(plot failed: {e})")


if __name__ == "__main__":
    main()

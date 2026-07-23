#!/usr/bin/env python3
"""
K=8 scale-selection heatmap: how does the canonical QWAB (s42_delta_detach, K=8,
with_null) choose among the 8 scales at each layer? Reference for the K=1 work.

Reads self._last_ctxscale_router_prob ([B,T,H,K] pi_scale, non-null selection mass)
per PaTHAttention layer after a forward pass, averages over batch/token/head ->
[layers x 8 scales], and plots a heatmap (x=scale rho, y=layer).
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
OUT = Path("/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat234_scale_card")
TARGET_LEN = int(sys.argv[1]) if len(sys.argv) > 1 else 512
N_SAMPLES = 8


def render(q, ctx):
    c = "".join(f"Title: {t}\n{' '.join(s).strip()}\n\n" for t, s in ctx)
    return f"Context:\n{c}\nQuestion: {q}\nAnswer:"


def main():
    dev = torch.device("cuda")
    cfg = AutoConfig.from_pretrained(CKPT)
    cfg.attn_implementation = "path_attn"
    cfg.wavelet_logit_bias_log_every = 10 ** 9
    cfg.wavelet_ctxscale_eval_log_once = False
    model = AutoModelForCausalLM.from_pretrained(CKPT, config=cfg, torch_dtype=torch.float32).to(dev).eval()

    layers = {}
    scales = None
    for m in model.modules():
        if isinstance(m, pa.PaTHAttention):
            _lid = getattr(m, "layer_idx", None)
            if _lid is None: continue
            lid = int(_lid)
            layers[lid] = m
            if scales is None and hasattr(m, "wavelet_ctxscale_scales"):
                scales = m.wavelet_ctxscale_scales.detach().float().cpu().numpy()
    L = max(layers) + 1
    K = int(len(scales)) if scales is not None else 8
    print(f"layers={L} K={K} scales={scales}")

    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    exs = []
    with open(HOTPOT) as f:
        for line in f:
            ex = json.loads(line)
            if ex["meta"]["target_total_tokens"] == 4096:
                exs.append(ex)
            if len(exs) >= N_SAMPLES: break

    acc = np.zeros((L, K)); nacc = np.zeros((L, K)); null_acc = np.zeros(L); null_n = np.zeros(L)
    for ex in exs:
        ids = tok(render(ex["question"], ex["context"]), return_tensors="pt", truncation=False)["input_ids"][:, :TARGET_LEN].to(dev)
        with torch.no_grad():
            model(input_ids=ids)
        for lid, m in layers.items():
            rp = getattr(m, "_last_ctxscale_router_prob", None)   # [B,T,H,K] or [B,T,K]
            if rp is None: continue
            r = rp.detach().float()
            while r.dim() > 1:            # collapse batch/token/head -> mean over all but last (K)
                r = r.mean(dim=0)
            acc[lid] += r.cpu().numpy(); nacc[lid] += 1
            nm = getattr(m, "_last_ctxscale_null_mass", None)
            if nm is not None:
                null_acc[lid] += float(nm.detach().float().mean().item()); null_n[lid] += 1

    mat = np.divide(acc, np.maximum(nacc, 1))     # [L,K] mean pi_scale per layer
    null_mean = np.divide(null_acc, np.maximum(null_n, 1))

    print("\n=== per-layer pi_scale (non-null selection weight) ===")
    hdr = "layer | " + " ".join(f"r{int(s):>6d}" for s in scales) + " | null | sum_g"
    print(hdr)
    for lid in range(L):
        row = " ".join(f"{mat[lid,j]:6.3f}" for j in range(K))
        print(f"  L{lid:>2} | {row} | {null_mean[lid]:.3f} | {mat[lid].sum():.3f}")

    # heatmap
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 6))
        im = ax.imshow(mat, aspect="auto", cmap="viridis", origin="upper")
        ax.set_xticks(range(K)); ax.set_xticklabels([f"ρ{int(s)}" for s in scales], rotation=45, ha="right")
        ax.set_yticks(range(L)); ax.set_yticklabels([f"L{i}" for i in range(L)])
        ax.set_xlabel("scale ρ"); ax.set_ylabel("layer")
        ax.set_title(f"K=8 QWAB (s42_delta_detach): per-layer scale selection  π_scale  @ L{TARGET_LEN}")
        for i in range(L):
            for j in range(K):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        color="white" if mat[i,j] < mat.max()*0.6 else "black", fontsize=7)
        fig.colorbar(im, ax=ax, label="π_scale (non-null selection weight)")
        out = OUT / f"k8_scale_selection_heatmap_L{TARGET_LEN}.png"
        fig.tight_layout(); fig.savefig(out, dpi=150)
        print(f"\nsaved: {out}")
    except Exception as e:
        print(f"(plot failed: {e})")


if __name__ == "__main__":
    main()

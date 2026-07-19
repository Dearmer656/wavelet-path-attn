#!/usr/bin/env python3
"""
PAT-225 K=4 seed-variance investigation: compare the learned router's
scale-selection distribution between s42 (F1=0.7038) and s43 (F1=0.6446)
on real HotpotQA-Long L4096 inputs, binned by query position, to see
whether/where the two seeds' routing behavior diverges.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/project/nlp-work5/hongyu-s/transformers/src")
sys.path.insert(0, "/project/nlp-work5/hongyu-s/flash-linear-attention")

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

HOTPOT_JSONL = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/hotpot_long_dev_uniform.jsonl"
N_SAMPLES = 24
TARGET_LEN = 4096
N_POS_BUCKETS = 8

CKPTS = {
    "s42": "/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat225_scale_card/S4_s42/checkpoint-15000",
    "s43": "/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat225_scale_card/S4_s43/checkpoint-15000",
}


def render_doc(title, sentences):
    body = " ".join(sentences).strip()
    return f"Title: {title}\n{body}\n\n"


def render_input(question, context):
    ctx = "".join(render_doc(t, s) for t, s in context)
    return f"Context:\n{ctx}\nQuestion: {question}\nAnswer:"


def load_examples():
    exs = []
    with open(HOTPOT_JSONL) as f:
        for line in f:
            ex = json.loads(line)
            if ex["meta"]["target_total_tokens"] == TARGET_LEN:
                exs.append(ex)
            if len(exs) >= N_SAMPLES:
                break
    return exs


def load_model(ckpt_dir, device):
    config = AutoConfig.from_pretrained(ckpt_dir)
    config.attn_implementation = "path_attn"
    config.wavelet_logit_bias_log_every = 10**9
    config.wavelet_ctxscale_eval_log_once = False
    model = AutoModelForCausalLM.from_pretrained(ckpt_dir, config=config, torch_dtype=torch.float32).to(device)
    model.eval()
    return model


def attach_hooks(model):
    captured = {}
    handles = []

    def make_hook(layer_idx):
        def hook(module, inp, out):
            logits = out.detach().float()  # [B, T, K+1] or [B, T, G, K+1]
            tau = 1.0
            g = torch.sigmoid(logits[..., 1:] / tau)
            sum_g = g.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            w = g / sum_g
            g0_gate = torch.sigmoid(logits[..., 0:1] / tau)
            pi_scale = g0_gate * w
            pi_null = (1.0 - g0_gate).clamp(0, 1)
            if pi_scale.dim() == 4:  # [B,T,G,K] -> average groups
                pi_scale = pi_scale.mean(dim=2)
                pi_null = pi_null.mean(dim=2)
            captured.setdefault(layer_idx, []).append({
                "pi_scale": pi_scale.squeeze(0).cpu(),  # [T, K]
                "pi_null": pi_null.squeeze(0).cpu(),    # [T, 1]
            })
        return hook

    for name, module in model.named_modules():
        if name.endswith("wavelet_ctx_router") and isinstance(module, torch.nn.Linear):
            parts = name.split(".")
            layer_idx = None
            for i, p in enumerate(parts):
                if p == "h" and i + 1 < len(parts):
                    layer_idx = int(parts[i + 1])
            if layer_idx is not None:
                handles.append(module.register_forward_hook(make_hook(layer_idx)))
    return captured, handles


def run_seed(tag, ckpt_dir, examples, tokenizer, device):
    print(f"=== Loading {tag} from {ckpt_dir} ===", flush=True)
    model = load_model(ckpt_dir, device)
    K = None
    for name, module in model.named_modules():
        if name.endswith("wavelet_ctx_router") and isinstance(module, torch.nn.Linear):
            K = module.out_features - 1
            break
    print(f"{tag}: K={K}")

    captured, handles = attach_hooks(model)

    per_example_layer_pos_pi = []  # list of dict[layer] -> [T_actual, K]
    for ei, ex in enumerate(examples):
        captured.clear()
        prompt = render_input(ex["question"], ex["context"])
        enc = tokenizer(prompt, return_tensors="pt", truncation=False)
        input_ids = enc["input_ids"].to(device)
        T = input_ids.shape[1]
        with torch.no_grad():
            model(input_ids=input_ids)
        per_example_layer_pos_pi.append({
            lid: captured[lid][0]["pi_scale"] for lid in captured
        })
        print(f"  [{tag}] example {ei+1}/{len(examples)} T={T} layers_captured={sorted(captured.keys())}", flush=True)

    for h in handles:
        h.remove()
    del model
    torch.cuda.empty_cache()
    return per_example_layer_pos_pi, K


def bucketize(per_example_layer_pos_pi, K, n_buckets):
    """Aggregate to [layer][bucket] -> mean pi_scale [K]."""
    layer_ids = sorted(per_example_layer_pos_pi[0].keys())
    agg = {lid: [[] for _ in range(n_buckets)] for lid in layer_ids}
    for ex_data in per_example_layer_pos_pi:
        for lid in layer_ids:
            pi = ex_data[lid]  # [T, K]
            T = pi.shape[0]
            edges = np.linspace(0, T, n_buckets + 1).astype(int)
            for b in range(n_buckets):
                lo, hi = edges[b], edges[b + 1]
                if hi > lo:
                    agg[lid][b].append(pi[lo:hi].mean(dim=0))
    out = {}
    for lid in layer_ids:
        out[lid] = []
        for b in range(n_buckets):
            if agg[lid][b]:
                stacked = torch.stack(agg[lid][b], dim=0)  # [n_ex, K]
                out[lid].append(stacked.mean(dim=0).numpy().tolist())
            else:
                out[lid].append([float("nan")] * K)
    return out


def main():
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples = load_examples()
    print(f"Loaded {len(examples)} examples at target_len={TARGET_LEN}")

    results = {}
    Ks = {}
    for tag, ckpt in CKPTS.items():
        per_ex, K = run_seed(tag, ckpt, examples, tokenizer, device)
        results[tag] = bucketize(per_ex, K, N_POS_BUCKETS)
        Ks[tag] = K

    out = {"K": Ks, "n_pos_buckets": N_POS_BUCKETS, "target_len": TARGET_LEN,
           "n_examples": len(examples), "results": results}
    out_path = Path(__file__).parent / "results" / "pat225_K4_scale_seedcompare.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")

    # Print a compact diff table: per layer, per bucket, L1 distance + which scale dominates
    K4 = Ks["s42"]
    layer_ids = sorted(results["s42"].keys())
    print("\n=== Per-layer, per-position-bucket scale distribution diff (s42 vs s43) ===")
    for lid in layer_ids:
        for b in range(N_POS_BUCKETS):
            p42 = np.array(results["s42"][lid][b])
            p43 = np.array(results["s43"][lid][b])
            l1 = float(np.abs(p42 - p43).sum())
            dom42 = int(np.argmax(p42))
            dom43 = int(np.argmax(p43))
            flag = " <== DIVERGE" if dom42 != dom43 and l1 > 0.15 else ""
            print(f"L{lid:2d} bucket{b} (pos~{int(b*TARGET_LEN/N_POS_BUCKETS)}-{int((b+1)*TARGET_LEN/N_POS_BUCKETS)}): "
                  f"s42={np.round(p42,3)} (dom=s{dom42}) s43={np.round(p43,3)} (dom=s{dom43}) L1={l1:.3f}{flag}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
PAT-230-B: does the QWAB router's scale mixture track the actual distance to the
gold-supporting evidence span, or is it position-agnostic?

For each HotpotQA-Long L4096 example we know the exact token span of the gold
support doc(s) (meta.support_start_tok / support_end_tok). We hook the router at
the LAST query position (right before "Answer:", i.e. where the model must
retrieve the support span) and compute an "effective log-scale" =
sum_k pi_scale[k] * log2(s_k) / sum(pi_scale) per layer. If routing is
retrieval-aware, this should correlate with log2(delta_gold), where
delta_gold = query_pos - support_end_tok.
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
N_SAMPLES = 80
TARGET_LEN = 4096

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
            if ex["meta"]["target_total_tokens"] == TARGET_LEN and ex["meta"].get("support_end_tok") is not None:
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


def get_effective_scales(model, K):
    max_exp = float(getattr(model.config, "wavelet_ctxscale_scale_max_exp", 14.0))
    if K == 1:
        return [2.0 ** (max_exp / 2.0)]
    return [2.0 ** (max_exp * i / (K - 1)) for i in range(K)]


def attach_hooks(model):
    captured = {}
    handles = []

    def make_hook(layer_idx):
        def hook(module, inp, out):
            logits = out.detach().float()
            g = torch.sigmoid(logits[..., 1:])
            sum_g = g.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            w = g / sum_g
            g0_gate = torch.sigmoid(logits[..., 0:1])
            pi_scale = g0_gate * w
            if pi_scale.dim() == 4:
                pi_scale = pi_scale.mean(dim=2)
            captured[layer_idx] = pi_scale.squeeze(0).cpu()  # [T, K]
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
    print(f"=== Loading {tag} ===", flush=True)
    model = load_model(ckpt_dir, device)
    K = int(getattr(model.config, "wavelet_ctxscale_k", 8))
    scales = get_effective_scales(model, K)
    log2_scales = np.log2(np.array(scales))
    print(f"{tag}: K={K} scales={scales}")

    # support_start_tok/support_end_tok in meta are relative to the concatenated
    # context docs only (make_hotpot_long.py's context_tok_cursor), NOT to the
    # full "Context:\n{ctx}\nQuestion:...\nAnswer:" prompt used for tokenization
    # here. Must add the exact "Context:\n" prefix token length before comparing
    # against query_pos, which IS in full-prompt token space.
    context_prefix_len = len(tokenizer("Context:\n", add_special_tokens=True)["input_ids"])
    print(f"{tag}: context_prefix_len={context_prefix_len} tokens")

    captured, handles = attach_hooks(model)

    records = []  # per example: {layer: eff_log_scale}, delta_gold
    for ei, ex in enumerate(examples):
        captured.clear()
        prompt = render_input(ex["question"], ex["context"])
        enc = tokenizer(prompt, return_tensors="pt", truncation=False)
        input_ids = enc["input_ids"].to(device)
        T = input_ids.shape[1]
        with torch.no_grad():
            model(input_ids=input_ids)

        support_end_tok = int(ex["meta"]["support_end_tok"])
        # Convert support_end_tok from context-relative to full-prompt-relative
        # token space before comparing against query_pos (see prefix-offset note
        # in run_seed above).
        support_end_tok_full = support_end_tok + context_prefix_len
        query_pos = T - 1
        delta_gold = max(1, query_pos - support_end_tok_full)

        layer_eff = {}
        for lid, pi in captured.items():
            row = pi[min(query_pos, pi.shape[0] - 1)].numpy()  # [K]
            s = row.sum()
            if s > 1e-6:
                eff_log_scale = float((row * log2_scales).sum() / s)
            else:
                eff_log_scale = float("nan")
            layer_eff[int(lid)] = eff_log_scale

        records.append({
            "example_id": ex.get("_id", ei),
            "delta_gold": delta_gold,
            "log2_delta_gold": float(np.log2(delta_gold)),
            "layer_eff_log_scale": layer_eff,
        })
        print(f"  [{tag}] ex {ei+1}/{len(examples)} T={T} delta_gold={delta_gold}", flush=True)

    for h in handles:
        h.remove()
    del model
    torch.cuda.empty_cache()
    return records, K, scales


def main():
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples = load_examples()
    print(f"Loaded {len(examples)} examples with known support span at L={TARGET_LEN}")

    all_results = {}
    for tag, ckpt in CKPTS.items():
        records, K, scales = run_seed(tag, ckpt, examples, tokenizer, device)
        all_results[tag] = {"K": K, "scales": scales, "records": records}

    out_path = Path(__file__).parent / "results" / "pat230_B_router_vs_gold_position.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {out_path}")

    print("\n=== Per-layer Pearson correlation: router effective log-scale vs log2(delta_gold) ===")
    for tag in all_results:
        recs = all_results[tag]["records"]
        layer_ids = sorted(recs[0]["layer_eff_log_scale"].keys())
        x = np.array([r["log2_delta_gold"] for r in recs])
        print(f"--- {tag} ---")
        for lid in layer_ids:
            y = np.array([r["layer_eff_log_scale"][lid] for r in recs])
            mask = np.isfinite(y) & np.isfinite(x)
            if mask.sum() < 5:
                print(f"  L{lid:2d}: insufficient data")
                continue
            corr = float(np.corrcoef(x[mask], y[mask])[0, 1])
            print(f"  L{lid:2d}: r={corr:+.3f} (n={int(mask.sum())})")


if __name__ == "__main__":
    main()

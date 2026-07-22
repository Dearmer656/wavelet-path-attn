#!/usr/bin/env python3
"""
PAT-225 K=1 per-scale backbone-response probe (divide-and-conquer).

For each K=1 single-scale model (scale 1/16/128/1024/16384), measure how much
the backbone ACTIVATES vs REJECTS that single scale, per layer, at each saved
checkpoint. For K=1 the router is [null, scale]; the non-null (scale-active)
mass is g0 = sigmoid(router_logits[...,0]). scale_active_mass -> 1 means the
backbone commits to USING the scale; -> 0 means it turns it OFF (pi_null->1).

This resolves the pi_top1 ambiguity from the training logs (decisive-toward-use
vs decisive-toward-reject) and tells us which single scales the backbone finds
valuable — the mechanistic definition of a "valuable scale" for the later
centered K>1 sweep. Measured on real HotpotQA-Long L4096 inputs.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/project/nlp-work5/hongyu-s/transformers/src")
sys.path.insert(0, "/project/nlp-work5/hongyu-s/flash-linear-attention")

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

HOTPOT_JSONL = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/hotpot_long_dev_uniform.jsonl"
N_SAMPLES = 12
TARGET_LEN = 4096
N_LAYERS = 12

BASE = "/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat225_scale_card"
# scale -> run dir (K=1 single-scale-location runs; 128 is the pre-existing center)
RUNS = {
    "scale_1": "S1_me0_s42",
    "scale_16": "S1_me8_s42",
    "scale_128": "S1_s42",
    "scale_1024": "S1_me20_s42",
    "scale_16384": "S1_me28_s42",
}


def render_input(question, context):
    ctx = "".join(f"Title: {t}\n{' '.join(s).strip()}\n\n" for t, s in context)
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


def discover_checkpoints(run_dir):
    steps = []
    for p in Path(run_dir).glob("checkpoint-*"):
        m = re.match(r"checkpoint-(\d+)$", p.name)
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def load_model(ckpt, device):
    cfg = AutoConfig.from_pretrained(ckpt)
    cfg.attn_implementation = "path_attn"
    cfg.wavelet_logit_bias_log_every = 10**9
    cfg.wavelet_ctxscale_eval_log_once = False
    m = AutoModelForCausalLM.from_pretrained(ckpt, config=cfg, torch_dtype=torch.float32).to(device)
    m.eval()
    return m


def attach_hooks(model, store):
    handles = []

    def make_hook(layer_idx):
        def hook(module, inp, out):
            logits = out.detach().float()
            # K=1 with_null: scale-active (non-null) mass = sigmoid(logits[...,0])
            g0 = torch.sigmoid(logits[..., 0])   # [...]  (drop last dim)
            if g0.dim() == 3:  # [B,T,H] head-wise
                g0 = g0.mean(dim=-1)
            store[layer_idx] = g0.squeeze(0).mean().item()  # mean over positions
        return hook

    for name, mod in model.named_modules():
        if name.endswith("wavelet_ctx_router") and isinstance(mod, torch.nn.Linear):
            parts = name.split(".")
            L = None
            for i, p in enumerate(parts):
                if p == "h" and i + 1 < len(parts):
                    L = int(parts[i + 1])
            if L is not None:
                handles.append(mod.register_forward_hook(make_hook(L)))
    return handles


def run_ckpt(ckpt, examples, tokenizer, device):
    model = load_model(ckpt, device)
    store = {}
    handles = attach_hooks(model, store)
    acc = {L: [] for L in range(N_LAYERS)}
    for ex in examples:
        store.clear()
        prompt = render_input(ex["question"], ex["context"])
        ids = tokenizer(prompt, return_tensors="pt", truncation=False)["input_ids"].to(device)
        with torch.no_grad():
            model(input_ids=ids)
        for L in range(N_LAYERS):
            if L in store:
                acc[L].append(store[L])
    for h in handles:
        h.remove()
    del model
    torch.cuda.empty_cache()
    return {L: (float(np.mean(v)) if v else None) for L, v in acc.items()}


def main():
    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    examples = load_examples()
    print(f"Loaded {len(examples)} L{TARGET_LEN} examples\n", flush=True)

    results = {}
    for scale, run in RUNS.items():
        steps = discover_checkpoints(f"{BASE}/{run}")
        if not steps:
            print(f"SKIP {scale} ({run}): no checkpoints", flush=True)
            continue
        results[scale] = {}
        for step in steps:
            print(f"=== {scale} step={step} ===", flush=True)
            results[scale][step] = run_ckpt(f"{BASE}/{run}/checkpoint-{step}", examples, tok, device)

    out = Path(__file__).parent / "results" / "pat225_k1_scale_response.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nSaved {out}")

    # summary: scale-active mass (mean over layers) per scale per step
    print("\n=== SCALE-ACTIVE MASS (mean sigmoid(router[...,0]); 1=backbone USES scale, 0=turns it OFF) ===")
    all_steps = sorted({s for sc in results.values() for s in sc})
    hdr = "  scale       " + "".join(f"| step{st:>6} " for st in all_steps)
    print(hdr)
    for scale in RUNS:
        if scale not in results:
            continue
        row = f"  {scale:<11} "
        for st in all_steps:
            v = results[scale].get(st)
            if v:
                m = float(np.mean([x for x in v.values() if x is not None]))
                row += f"|   {m:.4f}   "
            else:
                row += "|     -      "
        print(row)

    # per-layer at latest common step
    latest = max(all_steps)
    print(f"\n=== PER-LAYER scale-active mass @ step {latest} ===")
    print("  scale       " + "".join(f"L{L:>2}    " for L in range(N_LAYERS)))
    for scale in RUNS:
        r = results.get(scale, {}).get(latest)
        if not r:
            continue
        row = f"  {scale:<11} "
        for L in range(N_LAYERS):
            v = r.get(L) if r.get(L) is not None else r.get(str(L))
            row += f"{v:.3f} " if v is not None else "  -   "
        print(row)


if __name__ == "__main__":
    main()

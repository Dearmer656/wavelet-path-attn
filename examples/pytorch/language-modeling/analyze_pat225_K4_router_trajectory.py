#!/usr/bin/env python3
"""
PAT-225 K=4 outlier investigation: WHEN during training did s42's router policy
at L3/L7/L10 diverge from s43/s44? Sweeps checkpoints 5000/7500/10000/12500/15000
for all 3 seeds, computing mean pi_scale at those 3 layers, to localize the point
of divergence rather than only comparing final checkpoints.
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
N_SAMPLES = 16
TARGET_LEN = 4096
LAYERS_OF_INTEREST = (3, 7, 10)
STEPS = (5000, 7500, 10000, 12500, 15000)
SEEDS = ("s42", "s43", "s44")

BASE_DIR = "/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat225_scale_card"


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
            if layer_idx not in LAYERS_OF_INTEREST:
                return
            logits = out.detach().float()
            g = torch.sigmoid(logits[..., 1:])
            sum_g = g.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            w = g / sum_g
            g0_gate = torch.sigmoid(logits[..., 0:1])
            pi_scale = g0_gate * w
            if pi_scale.dim() == 4:
                pi_scale = pi_scale.mean(dim=2)
            captured[layer_idx] = pi_scale.squeeze(0).mean(dim=0).cpu().numpy()  # [K], mean over positions
        return hook

    for name, module in model.named_modules():
        if name.endswith("wavelet_ctx_router") and isinstance(module, torch.nn.Linear):
            parts = name.split(".")
            layer_idx = None
            for i, p in enumerate(parts):
                if p == "h" and i + 1 < len(parts):
                    layer_idx = int(parts[i + 1])
            if layer_idx is not None and layer_idx in LAYERS_OF_INTEREST:
                handles.append(module.register_forward_hook(make_hook(layer_idx)))
    return captured, handles


def run_one(ckpt_dir, examples, tokenizer, device):
    model = load_model(ckpt_dir, device)
    captured, handles = attach_hooks(model)
    per_layer_acc = {L: [] for L in LAYERS_OF_INTEREST}
    for ex in examples:
        captured.clear()
        prompt = render_input(ex["question"], ex["context"])
        enc = tokenizer(prompt, return_tensors="pt", truncation=False)
        input_ids = enc["input_ids"].to(device)
        with torch.no_grad():
            model(input_ids=input_ids)
        for L in LAYERS_OF_INTEREST:
            if L in captured:
                per_layer_acc[L].append(captured[L])
    for h in handles:
        h.remove()
    del model
    torch.cuda.empty_cache()
    out = {}
    for L in LAYERS_OF_INTEREST:
        if per_layer_acc[L]:
            out[L] = np.stack(per_layer_acc[L], axis=0).mean(axis=0).tolist()
    return out


def main():
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    examples = load_examples()
    print(f"Loaded {len(examples)} examples at target_len={TARGET_LEN}")

    results = {}
    for seed in SEEDS:
        results[seed] = {}
        for step in STEPS:
            ckpt = f"{BASE_DIR}/S4_{seed}/checkpoint-{step}"
            if not Path(ckpt).is_dir():
                print(f"SKIP {seed} step={step}: no checkpoint at {ckpt}")
                continue
            print(f"=== {seed} step={step} ===", flush=True)
            out = run_one(ckpt, examples, tokenizer, device)
            results[seed][step] = out
            for L in LAYERS_OF_INTEREST:
                if L in out:
                    print(f"  L{L}: {np.round(out[L], 3)}", flush=True)

    out_path = Path(__file__).parent / "results" / "pat225_K4_router_trajectory.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")

    print("\n=== Divergence summary: |s42 - s43| and |s42 - s44| per layer per step ===")
    for L in LAYERS_OF_INTEREST:
        print(f"--- L{L} ---")
        for step in STEPS:
            if step not in results["s42"] or L not in results["s42"].get(step, {}):
                continue
            v42 = np.array(results["s42"][step][L])
            row = f"  step={step:>6}: s42={np.round(v42,3)}"
            for other in ("s43", "s44"):
                if step in results[other] and L in results[other].get(step, {}):
                    vo = np.array(results[other][step][L])
                    d = float(np.abs(v42 - vo).sum())
                    row += f"  |s42-{other}|={d:.3f}"
            print(row)


if __name__ == "__main__":
    main()

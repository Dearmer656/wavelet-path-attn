#!/usr/bin/env python3
"""
PAT-225 follow-up: does the router's dominant scale (argmax of pi_scale)
change identity more than once over training ("reversal"), rather than
just monotonically sharpening/flattening? Extends
analyze_pat225_K4_router_trajectory.py to (a) all 12 layers instead of
just L3/L7/L10, and (b) three runs sharing seed=42's recipe: the
original best run (S4_s42, F1=0.7038), the hardware-rerun (S4_s42_rerun,
F1=0.6694), and the zero-init router validation (S4_s42_zeroinit, still
training). Sweeps every checkpoint that actually exists on disk for each
run rather than a fixed step list.
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
N_SAMPLES = 16
TARGET_LEN = 4096
N_LAYERS = 12

BASE_DIR = "/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat225_scale_card"
RUNS = {
    "orig_s42": f"{BASE_DIR}/S4_s42",
    "rerun_s42": f"{BASE_DIR}/S4_s42_rerun",
    "zeroinit_s42": f"{BASE_DIR}/S4_s42_zeroinit",
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


def discover_checkpoints(run_dir):
    steps = []
    for p in Path(run_dir).glob("checkpoint-*"):
        m = re.match(r"checkpoint-(\d+)$", p.name)
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


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
            if layer_idx is not None:
                handles.append(module.register_forward_hook(make_hook(layer_idx)))
    return captured, handles


def run_one(ckpt_dir, examples, tokenizer, device):
    model = load_model(ckpt_dir, device)
    captured, handles = attach_hooks(model)
    per_layer_acc = {L: [] for L in range(N_LAYERS)}
    for ex in examples:
        captured.clear()
        prompt = render_input(ex["question"], ex["context"])
        enc = tokenizer(prompt, return_tensors="pt", truncation=False)
        input_ids = enc["input_ids"].to(device)
        with torch.no_grad():
            model(input_ids=input_ids)
        for L in range(N_LAYERS):
            if L in captured:
                per_layer_acc[L].append(captured[L])
    for h in handles:
        h.remove()
    del model
    torch.cuda.empty_cache()
    out = {}
    for L in range(N_LAYERS):
        if per_layer_acc[L]:
            out[L] = np.stack(per_layer_acc[L], axis=0).mean(axis=0).tolist()
    return out


def find_reversals(step_to_vec):
    """A 'reversal' = the argmax scale index changes identity more than
    once across the sorted step sequence (i.e. A->B->A or A->B->C->B etc,
    not just a single one-time rank swap that then stays put)."""
    steps = sorted(step_to_vec.keys())
    argmaxes = [int(np.argmax(step_to_vec[s])) for s in steps]
    distinct_switch_points = []
    for i in range(1, len(argmaxes)):
        if argmaxes[i] != argmaxes[i - 1]:
            distinct_switch_points.append((steps[i], argmaxes[i - 1], argmaxes[i]))
    # genuine reversal: some scale index appears, disappears, and reappears
    # as argmax non-contiguously.
    seen_runs = []
    for a in argmaxes:
        if not seen_runs or seen_runs[-1] != a:
            seen_runs.append(a)
    reversal = len(set(seen_runs)) < len(seen_runs)  # an index recurs after being displaced
    return argmaxes, distinct_switch_points, reversal


def main():
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    examples = load_examples()
    print(f"Loaded {len(examples)} examples at target_len={TARGET_LEN}", flush=True)

    results = {}
    for run_name, run_dir in RUNS.items():
        steps = discover_checkpoints(run_dir)
        print(f"=== {run_name}: checkpoints found = {steps} ===", flush=True)
        results[run_name] = {}
        for step in steps:
            ckpt = f"{run_dir}/checkpoint-{step}"
            print(f"--- {run_name} step={step} ---", flush=True)
            out = run_one(ckpt, examples, tokenizer, device)
            results[run_name][step] = out

    out_path = Path(__file__).parent / "results" / "pat225_scale_reversal.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")

    print("\n=== Reversal scan (argmax scale index per layer, per run, across steps) ===")
    for run_name in RUNS:
        if run_name not in results or not results[run_name]:
            continue
        print(f"\n--- {run_name} ---")
        steps = sorted(results[run_name].keys())
        for L in range(N_LAYERS):
            step_to_vec = {s: results[run_name][s][str(L) if str(L) in results[run_name][s] else L]
                           for s in steps if (str(L) in results[run_name][s] or L in results[run_name][s])}
            if len(step_to_vec) < 3:
                continue
            argmaxes, switches, reversal = find_reversals(step_to_vec)
            flag = "  <-- REVERSAL" if reversal else ""
            print(f"  L{L:>2}: steps={steps} argmax_seq={argmaxes} switches={switches}{flag}")


if __name__ == "__main__":
    main()

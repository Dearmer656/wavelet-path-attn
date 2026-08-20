#!/usr/bin/env python3
"""Does PA's own (pre-QWAB) attention logit behave differently for key-query
distances within L_train vs beyond L_train?

Forwards a real postfix checkpoint on real HotpotQA-Long data at L2048,
captures the PA-only logits (_last_logits_pa_only, i.e. E_base_raw) per layer,
and compares statistics (mean, std, and within-region softmax entropy) for
distance < L_TRAIN vs distance >= L_TRAIN.
"""
import sys
import json
import numpy as np
import torch

sys.path.insert(0, "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling")
import run_clm  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

N_LAYERS = 12
EVAL_T = 2048
N_EXAMPLES = 2
L_TRAIN = 512

CHECKPOINT = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat225_postfix_docorder/K1_rho256_ricker_s42/checkpoint-15000"
CFG_PATH = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat225_postfix_docorder/K1_rho256_ricker_s42/supply_model.cfg"


def load_model(checkpoint_dir, cfg_path, block_size):
    config = AutoConfig.from_pretrained(checkpoint_dir)
    cfg = run_clm.read_kv_config(str(cfg_path))
    run_clm.add_missing_to_hf_config(config, cfg)
    config.attn_implementation = "path_attn"
    config.path_attn_impl = "pytorch"
    config.use_cache = False
    config.block_size = block_size
    return AutoModelForCausalLM.from_pretrained(checkpoint_dir, config=config)


def build_batch(tokenizer, length, n):
    texts = []
    with open("/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/hotpot_long_dev_uniform.jsonl") as f:
        for line in f:
            if len(texts) >= n:
                break
            ex = json.loads(line)
            ctx = " ".join(f"{c[0]}: {c[1]}" for c in ex["context"])
            text = f"Context:\n{ctx}\n\nQuestion: {ex['question']}\nAnswer: {ex['answer']}"
            texts.append(text)
    rows = []
    for t in texts:
        ids = tokenizer(t, add_special_tokens=True, truncation=True, max_length=length)["input_ids"]
        if len(ids) < length:
            ids = ids + [tokenizer.eos_token_id] * (length - len(ids))
        rows.append(ids[:length])
    return torch.tensor(rows, dtype=torch.long)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    batch = build_batch(tokenizer, EVAL_T, N_EXAMPLES).to(device)

    model = load_model(CHECKPOINT, CFG_PATH, EVAL_T)
    model.eval().to(device)
    with torch.no_grad():
        model(input_ids=batch)

    q_pos = torch.arange(EVAL_T, device=device).view(-1, 1)
    k_pos = torch.arange(EVAL_T, device=device).view(1, -1)
    dist = q_pos - k_pos  # [T,T]
    causal_mask = dist >= 0
    near_mask = causal_mask & (dist < L_TRAIN)
    far_mask = causal_mask & (dist >= L_TRAIN)

    print(f"{'layer':>5} {'near_mean':>10} {'near_std':>9} {'far_mean':>10} {'far_std':>9} {'near_H_mean':>11} {'far_H_mean':>10}")
    for layer in range(N_LAYERS):
        core = getattr(model.transformer.h[layer].attn, "core", model.transformer.h[layer].attn)
        pa_logits = getattr(core, "_last_logits_pa_only", None)
        if pa_logits is None:
            print(f"{layer:>5}  MISSING HOOK")
            continue
        # pa_logits: [B,H,T,T] presumably; average over batch/heads for a per-position summary
        pl = pa_logits.detach().float()
        if pl.dim() == 4:
            pl_bh = pl.reshape(-1, pl.shape[-2], pl.shape[-1])  # [B*H, T, T]
        else:
            pl_bh = pl.unsqueeze(0)

        near_vals = pl_bh[:, near_mask].reshape(-1)
        far_vals = pl_bh[:, far_mask].reshape(-1)

        # within-region softmax entropy per query row (only over keys in that region)
        near_entropies = []
        far_entropies = []
        sample_rows = list(range(L_TRAIN, EVAL_T, 200))  # rows that actually have both near and far keys
        for r in sample_rows:
            row_near = pl_bh[:, r, :][:, near_mask[r]]
            row_far = pl_bh[:, r, :][:, far_mask[r]]
            if row_near.numel() > 0:
                p = torch.softmax(row_near, dim=-1)
                h = -(p * (p.clamp_min(1e-12)).log()).sum(dim=-1)
                near_entropies.append(h.mean().item())
            if row_far.numel() > 0:
                p = torch.softmax(row_far, dim=-1)
                h = -(p * (p.clamp_min(1e-12)).log()).sum(dim=-1)
                far_entropies.append(h.mean().item())

        near_h_mean = float(np.mean(near_entropies)) if near_entropies else float("nan")
        far_h_mean = float(np.mean(far_entropies)) if far_entropies else float("nan")

        print(
            f"{layer:>5} {near_vals.mean().item():>10.4f} {near_vals.std().item():>9.4f} "
            f"{far_vals.mean().item():>10.4f} {far_vals.std().item():>9.4f} "
            f"{near_h_mean:>11.4f} {far_h_mean:>10.4f}"
        )


if __name__ == "__main__":
    main()

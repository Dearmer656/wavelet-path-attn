#!/usr/bin/env python3
"""One case, one layer, one head, one query row: plot PA-raw logit and within-row
softmax probability vs key position, near vs far region colored separately, with
the region entropies annotated -- to visually sanity-check the "far attention is
more peaked" finding against a single concrete row instead of an aggregate stat.
"""
import sys
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling")
import run_clm  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

EVAL_T = 2048
L_TRAIN = 512
LAYER = 6
HEAD = 0
QUERY_POS = 1800  # near keys: [1288,1800], far keys: [0,1288)

CHECKPOINT = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat225_postfix_docorder/PA_only_s42/checkpoint-15000"
CFG_PATH = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat225_postfix_docorder/PA_only_s42/supply_model.cfg"


def load_model(checkpoint_dir, cfg_path, block_size):
    config = AutoConfig.from_pretrained(checkpoint_dir)
    cfg = run_clm.read_kv_config(str(cfg_path))
    run_clm.add_missing_to_hf_config(config, cfg)
    config.attn_implementation = "path_attn"
    config.path_attn_impl = "pytorch"
    config.use_cache = False
    config.block_size = block_size
    return AutoModelForCausalLM.from_pretrained(checkpoint_dir, config=config)


def build_one_example(tokenizer, length):
    with open("/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/hotpot_long_dev_uniform.jsonl") as f:
        for line in f:
            ex = json.loads(line)
            if ex.get("meta", {}).get("target_total_tokens") == 2048:
                ctx = " ".join(f"{c[0]}: {c[1]}" for c in ex["context"])
                text = f"Context:\n{ctx}\n\nQuestion: {ex['question']}\nAnswer: {ex['answer']}"
                break
    ids = tokenizer(text, add_special_tokens=True, truncation=True, max_length=length)["input_ids"]
    if len(ids) < length:
        ids = ids + [tokenizer.eos_token_id] * (length - len(ids))
    return torch.tensor([ids[:length]], dtype=torch.long)


def entropy(probs):
    p = probs.clamp_min(1e-12)
    return float(-(p * p.log()).sum().item())


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    batch = build_one_example(tokenizer, EVAL_T).to(device)

    model = load_model(CHECKPOINT, CFG_PATH, EVAL_T)
    model.eval().to(device)
    with torch.no_grad():
        model(input_ids=batch)

    core = getattr(model.transformer.h[LAYER].attn, "core", model.transformer.h[LAYER].attn)
    logits = getattr(core, "_last_pa_raw_logits_unconditional", None)
    assert logits is not None, "hook missing"
    row = logits[0, HEAD, QUERY_POS, : QUERY_POS + 1].detach().float().cpu().numpy()  # causal keys only

    key_pos = np.arange(0, QUERY_POS + 1)
    near_mask = key_pos >= (QUERY_POS - L_TRAIN + 1) if QUERY_POS >= L_TRAIN else np.ones_like(key_pos, dtype=bool)
    # Match the session's near/far convention: dist = q - k; near: 0<=dist<L_TRAIN; far: dist>=L_TRAIN
    dist = QUERY_POS - key_pos
    near_mask = dist < L_TRAIN
    far_mask = ~near_mask

    row_t = torch.from_numpy(row)
    near_probs = torch.softmax(row_t[near_mask], dim=-1)
    far_probs = torch.softmax(row_t[far_mask], dim=-1) if far_mask.any() else None
    near_H = entropy(near_probs)
    far_H = entropy(far_probs) if far_probs is not None else float("nan")

    full_probs = torch.softmax(row_t, dim=-1).numpy()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax = axes[0]
    ax.plot(key_pos[far_mask], row[far_mask], color="tab:red", lw=0.8, label=f"far (dist>={L_TRAIN}), H={far_H:.3f}")
    ax.plot(key_pos[near_mask], row[near_mask], color="tab:blue", lw=0.8, label=f"near (dist<{L_TRAIN}), H={near_H:.3f}")
    ax.axvline(QUERY_POS - L_TRAIN, color="gray", ls="--", lw=1)
    ax.set_ylabel("PA raw logit")
    ax.set_title(f"layer={LAYER} head={HEAD} query_pos={QUERY_POS} (PA_only_s42, L2048 eval)")
    ax.legend()

    ax2 = axes[1]
    ax2.plot(key_pos[far_mask], full_probs[far_mask], color="tab:red", lw=0.8)
    ax2.plot(key_pos[near_mask], full_probs[near_mask], color="tab:blue", lw=0.8)
    ax2.axvline(QUERY_POS - L_TRAIN, color="gray", ls="--", lw=1)
    ax2.set_ylabel("full-row softmax prob")
    ax2.set_xlabel("key position")

    fig.tight_layout()
    out_path = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_postfix_4way_s42/one_row_logit_line.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    print(f"near_entropy={near_H:.4f} far_entropy={far_H:.4f} n_near={int(near_mask.sum())} n_far={int(far_mask.sum())}")


if __name__ == "__main__":
    main()

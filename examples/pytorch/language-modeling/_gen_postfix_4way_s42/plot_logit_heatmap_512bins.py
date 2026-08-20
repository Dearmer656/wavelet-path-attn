#!/usr/bin/env python3
"""L2048 attention-logit heatmap, binned into 512-token blocks (4x4 grid).

Loads postfix K1_rho256_ricker (QWAB, full post-bias logits) and PA_only
(PA-raw logits) at L_train=512, forwards real HotpotQA-Long data at T=2048,
averages logits within each (query-bin, key-bin) 512x512 block (respecting
the causal mask, including inside the diagonal block), averaged over layers,
heads, and batch. Saves a side-by-side PNG.
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

N_LAYERS = 12
EVAL_T = 2048
N_EXAMPLES = 8
BIN = 512
N_BINS = EVAL_T // BIN

CHECKPOINTS = {
    "K1_rho256_ricker (QWAB, post-bias logits)": (
        "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat225_postfix_docorder/K1_rho256_ricker_s42/checkpoint-15000",
        "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat225_postfix_docorder/K1_rho256_ricker_s42/supply_model.cfg",
        "wavelet",
    ),
    "PA_only (PA-raw logits)": (
        "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat225_postfix_docorder/PA_only_s42/checkpoint-15000",
        "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat225_postfix_docorder/PA_only_s42/supply_model.cfg",
        "pa_raw",
    ),
}

HOOK_BY_KIND = {
    "wavelet": "_last_logits_full",
    "pa_raw": "_last_pa_raw_logits_unconditional",
}


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


def binned_logit_matrix(logits_bh_t_t, causal_mask):
    """logits_bh_t_t: [N, T, T] (batch*head flattened). Returns [N_BINS, N_BINS] mean over
    valid (causal) entries, averaged over N."""
    T = logits_bh_t_t.shape[-1]
    out = np.full((N_BINS, N_BINS), np.nan, dtype=np.float64)
    for qi in range(N_BINS):
        for ki in range(N_BINS):
            if ki > qi:
                continue  # fully non-causal block, leave NaN
            q0, q1 = qi * BIN, (qi + 1) * BIN
            k0, k1 = ki * BIN, (ki + 1) * BIN
            block = logits_bh_t_t[:, q0:q1, k0:k1]
            mask_block = causal_mask[q0:q1, k0:k1]
            if not mask_block.any():
                continue
            vals = block[:, mask_block]
            out[qi, ki] = vals.mean().item()
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    batch = build_batch(tokenizer, EVAL_T, N_EXAMPLES).to(device)

    q_pos = torch.arange(EVAL_T, device=device).view(-1, 1)
    k_pos = torch.arange(EVAL_T, device=device).view(1, -1)
    causal_mask = (q_pos - k_pos) >= 0

    results = {}
    for name, (ckpt, cfg_path, logit_kind) in CHECKPOINTS.items():
        print(f"Loading {name} ...")
        model = load_model(ckpt, cfg_path, EVAL_T)
        model.eval().to(device)
        with torch.no_grad():
            model(input_ids=batch)

        layer_mats = []
        for layer in range(N_LAYERS):
            core = getattr(model.transformer.h[layer].attn, "core", model.transformer.h[layer].attn)
            logits = getattr(core, HOOK_BY_KIND[logit_kind], None)
            if logits is None:
                print(f"  layer={layer} MISSING HOOK, skip")
                continue
            pl = logits.detach().float()
            if pl.dim() == 4:
                pl_bh = pl.reshape(-1, pl.shape[-2], pl.shape[-1])
            else:
                pl_bh = pl.unsqueeze(0)
            mat = binned_logit_matrix(pl_bh, causal_mask)
            layer_mats.append(mat)
        layer_mats = np.stack(layer_mats, axis=0)
        mean_mat = np.nanmean(layer_mats, axis=0)
        results[name] = mean_mat
        del model
        torch.cuda.empty_cache()

    fig, axes = plt.subplots(1, len(results), figsize=(6.5 * len(results), 5.5))
    if len(results) == 1:
        axes = [axes]
    tick_labels = [f"{i*BIN}-{(i+1)*BIN}" for i in range(N_BINS)]
    for ax, (name, mat) in zip(axes, results.items()):
        im = ax.imshow(mat, cmap="viridis", aspect="equal")
        ax.set_xticks(range(N_BINS))
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.set_yticks(range(N_BINS))
        ax.set_yticklabels(tick_labels)
        ax.set_xlabel("key position bin")
        ax.set_ylabel("query position bin")
        ax.set_title(name, fontsize=10)
        for qi in range(N_BINS):
            for ki in range(N_BINS):
                if not np.isnan(mat[qi, ki]):
                    ax.text(ki, qi, f"{mat[qi, ki]:.2f}", ha="center", va="center",
                             color="white" if mat[qi, ki] < np.nanmean(mat) else "black", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean logit")

    fig.suptitle(f"Mean attention logit by 512-token bin, L2048 eval, L_train=512 (avg over {N_LAYERS} layers, heads, {N_EXAMPLES} examples)")
    fig.tight_layout()
    out_path = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/_gen_postfix_4way_s42/logit_heatmap_512bins_L2048.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

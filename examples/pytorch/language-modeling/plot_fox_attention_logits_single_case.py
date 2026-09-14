#!/usr/bin/env python3
"""Visualize FoX's actual learned attention distribution (not just the retention-rate
proxy) for one real HotpotQA-Long L4096 example. Manually recomputes the exact
attention formula used by fla's parallel_forgetting_attn (confirmed by reading
fla/ops/attn/parallel.py): logits[i,j] = scale*(q_i . k_j) + g_cumsum[i] - g_cumsum[j],
causal, then standard softmax -- mathematically identical to the Triton kernel's
online-softmax computation, just done densely in PyTorch for one example so we can
actually look at the resulting [T] attention row for the LAST query position (right
before the model generates its answer).

Picks a few layers/heads spanning the retention spectrum already measured this session
(forget_gate_per_layer_perhead_L4096.json): a "star" near-persistent head, a "local"
low-retention head, and marks where the gold supporting-fact tokens and the question
sit in the assembled prompt, so we can see directly whether attention actually
concentrates on the evidence or just on recency.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/project/nlp-work5/hongyu-s/transformers/src")
sys.path.insert(0, "/project/nlp-work5/hongyu-s/flash-linear-attention")

import run_clm  # noqa: E402 -- reuse build_context_budgeted exactly as eval does

CKPT = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/mix_medium_owt_fox_10ep_s42_a6000x4/checkpoint-15000"
JSONL = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/hotpot_long_dev.jsonl"
TARGET_LEN = 4096
BLOCK_SIZE = 4096
OUT_PNG = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/results/fox_medium_s42_finetuned_ckpt15000_forgetgate/attention_logits_single_case_L4096.png"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
run_clm.GLOBAL_TOKENIZER = tokenizer  # build_context_budgeted's _get_tokenizer() reads this

config = AutoConfig.from_pretrained(CKPT)
model = AutoModelForCausalLM.from_pretrained(CKPT, config=config, torch_dtype=torch.float32)
model.to(device)
model.eval()

# ---- Load one real record and build the exact eval-time prompt ----
rec = None
with open(JSONL) as f:
    for line in f:
        r = json.loads(line)
        if r["meta"]["target_total_tokens"] == TARGET_LEN:
            rec = r
            break
assert rec is not None, f"no record found with target_total_tokens={TARGET_LEN}"

ex = {
    "question": rec["question"],
    "answer": rec["answer"],
    "supporting_facts": {
        "title": [sf[0] for sf in rec["supporting_facts"]],
        "sent_id": [sf[1] for sf in rec["supporting_facts"]],
    },
    "context": {"title": [c[0] for c in rec["context"]], "sentences": [c[1] for c in rec["context"]]},
}

tok = tokenizer
ctx_part = "Context:\n"
q_part = f"\nQuestion: {rec['question']}\n"
ans_prompt = "Answer:"
prefix_ids = tok(ctx_part, add_special_tokens=True, truncation=False)["input_ids"]
q_ids = tok(q_part, add_special_tokens=False, truncation=False)["input_ids"]
ans_prompt_ids = tok(ans_prompt, add_special_tokens=False, truncation=False)["input_ids"]
suffix_ids = q_ids + ans_prompt_ids
ans_ids = tok(" " + rec["answer"].strip(), add_special_tokens=False)["input_ids"]

budget_context = BLOCK_SIZE - len(prefix_ids) - len(suffix_ids) - len(ans_ids) - 1

ctx_text, ctx_ids, status = run_clm.build_context_budgeted(
    ex, budget_context, prefer_same_title=True, min_tokens=0, respect_doc_order=False,
)
assert ctx_ids is not None, f"build_context_budgeted failed: status={status}"

# insertion order default (respect_doc_order=False) puts supporting facts FIRST --
# recompute how many leading tokens of ctx_ids belong to supporting facts by re-deriving
# the same selection fla run_clm.py does internally (supporting sentences added before
# any distractor fill).
sf_pairs = run_clm._get_supporting_pairs(ex)
title2sents = run_clm._build_title2sents(ex)
newline_cost = len(tok("\n", add_special_tokens=False)["input_ids"])
n_evidence_tokens = 0
for i, (t, sid) in enumerate(sf_pairs):
    sents = title2sents.get(t, [])
    if 0 <= sid < len(sents):
        text = f"{t}: {sents[sid]}"
        ids = tok(text, add_special_tokens=False)["input_ids"]
        n_evidence_tokens += len(ids) + (newline_cost if i > 0 else 0)

prompt_ids = prefix_ids + ctx_ids + suffix_ids
question_start = len(prefix_ids) + len(ctx_ids)
evidence_end = len(prefix_ids) + n_evidence_tokens

print(f"prompt length (context part) = {len(prompt_ids)} tokens, "
      f"evidence = [{len(prefix_ids)}, {evidence_end}), question+answer-prompt = [{question_start}, {len(prompt_ids)})")

input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
T = input_ids.shape[1]

captured = {}


def make_hook(layer_idx):
    def hook(module, args, kwargs, output):
        if args:
            hidden_states = args[0]
        else:
            hidden_states = kwargs["hidden_states"]
        q = module.q_proj(hidden_states)
        k = module.k_proj(hidden_states)
        f = F.logsigmoid(module.f_proj(hidden_states).float())
        captured[layer_idx] = (q.detach(), k.detach(), f.detach(), module.head_dim, module.num_heads)
    return hook


hooks = []
for name, module in model.named_modules():
    if module.__class__.__name__ == "ForgettingAttention":
        lid = int(getattr(module, "layer_idx", -1))
        hooks.append(module.register_forward_hook(make_hook(lid), with_kwargs=True))

with torch.no_grad():
    model(input_ids=input_ids, attention_mask=None)

for h in hooks:
    h.remove()

print(f"Captured {len(captured)} layers.")


def attn_row_for_query(layer_idx, head_idx, query_pos):
    q, k, f, head_dim, num_heads = captured[layer_idx]
    B, T_, _ = q.shape
    q = q.view(B, T_, num_heads, head_dim)[0, :, head_idx, :]  # [T, D]
    k = k.view(B, T_, num_heads, head_dim)[0, :, head_idx, :]  # [T, D]
    g = f[0, :, head_idx]  # [T]
    g_cumsum = torch.cumsum(g, dim=0)  # [T]
    scale = head_dim ** -0.5
    q_i = q[query_pos]  # [D]
    logits = (k @ q_i) * scale + g_cumsum[query_pos] - g_cumsum  # [T]
    logits[query_pos + 1:] = float("-inf")  # causal
    probs = torch.softmax(logits, dim=0)
    return probs.cpu().numpy()


# Pick a handful of (layer, head) combos spanning the retention spectrum, using the
# per-head JSON already produced this session.
FG_JSON = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/results/fox_medium_s42_finetuned_ckpt15000_forgetgate/forget_gate_per_layer_perhead_L4096.json"
with open(FG_JSON) as fjson:
    fg = json.load(fjson)

picks = []
for lid_str in ["0", "4", "8", "17", "21"]:
    heads = fg["per_layer_per_head_mean_retention"][lid_str]
    max_h = max(range(len(heads)), key=lambda h: heads[h])
    min_h = min(range(len(heads)), key=lambda h: heads[h])
    picks.append((int(lid_str), max_h, f"L{lid_str} star-head(h{max_h}) r={heads[max_h]:.3f}"))
    picks.append((int(lid_str), min_h, f"L{lid_str} local-head(h{min_h}) r={heads[min_h]:.3f}"))

query_pos = T - 2  # last real token before EOS (i.e. right at/after "Answer:")

fig, axes = plt.subplots(len(picks), 1, figsize=(12, 2.2 * len(picks)), sharex=True)
for ax, (lid, hid, label) in zip(axes, picks):
    probs = attn_row_for_query(lid, hid, query_pos)
    ax.plot(probs, linewidth=0.7, color="#2b6cb0")
    ax.axvspan(len(prefix_ids), evidence_end, color="#e53e3e", alpha=0.25, label="gold evidence")
    ax.axvspan(question_start, T, color="#38a169", alpha=0.2, label="question+answer-prompt")
    ax.set_ylabel(label, fontsize=8)
    ax.set_yscale("log")

axes[0].legend(loc="upper right", fontsize=7)
axes[-1].set_xlabel("Key position")
fig.suptitle(
    f"FoX medium (finetuned) -- actual attention weights from the last query position\n"
    f"HotpotQA-Long L4096 real example, red=gold evidence tokens, green=question/answer-prompt tokens",
    fontsize=10,
)
fig.tight_layout(rect=[0, 0, 1, 0.94])
os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
fig.savefig(OUT_PNG, dpi=150)
print(f"Wrote {OUT_PNG}")

# Also print summary stats: total attention mass on evidence vs question vs everything else
summary = {}
for lid, hid, label in picks:
    probs = attn_row_for_query(lid, hid, query_pos)
    mass_evidence = float(probs[len(prefix_ids):evidence_end].sum())
    mass_question = float(probs[question_start:].sum())
    mass_other = float(1.0 - mass_evidence - mass_question)
    summary[label] = {"mass_evidence": mass_evidence, "mass_question": mass_question, "mass_other": mass_other}
    print(f"{label}: evidence={mass_evidence:.4f} question={mass_question:.4f} other={mass_other:.4f}")

with open(OUT_PNG.replace(".png", "_summary.json"), "w") as fout:
    json.dump(summary, fout, indent=2)

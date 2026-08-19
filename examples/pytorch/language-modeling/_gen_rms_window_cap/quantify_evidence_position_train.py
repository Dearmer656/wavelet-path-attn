#!/usr/bin/env python3
"""Quantify where evidence (supporting-fact sentences) actually lands in the
final assembled training prompt, using the REAL run_clm.py logic
(build_hotpot_qa / build_context_budgeted), our actual training cfg
(hotpot_question_position='later', block_size=512), and the REAL data source
training uses (raw HF-hub hotpot_qa distractor train split, NOT the custom
uniform jsonl -- confirmed no --hotpot_long_jsonl in any train script).
"""
import sys
from transformers import GPT2Tokenizer
from datasets import load_dataset

BLOCK_SIZE = 512
N_SAMPLES = 500

tok = GPT2Tokenizer.from_pretrained("gpt2")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def token_ids(text, add_special=False):
    return tok(text, add_special_tokens=add_special, truncation=False)["input_ids"]


def build_title2sents(ex):
    title2sents = {}
    ctx = ex["context"]
    for t, sents in zip(ctx["title"], ctx["sentences"]):
        title2sents[str(t)] = sents
    return title2sents


def get_supporting_pairs(ex):
    sf = ex["supporting_facts"]
    return [(str(t), int(i)) for t, i in zip(sf["title"], sf["sent_id"])]


def build_context_budgeted(ex, budget_tokens, prefer_same_title=True):
    """Faithful reimplementation of run_clm.py:3237, using the REAL tokenizer."""
    title2sents = build_title2sents(ex)
    newline_ids = token_ids("\n")
    newline_cost = len(newline_ids)
    sf_pairs = get_supporting_pairs(ex)
    if not sf_pairs:
        return None, "no_supporting"

    selected = []
    selected_keys = set()
    total = 0

    def add_sent(t, sid, is_evidence):
        nonlocal total
        sents = title2sents.get(t, [])
        if not (0 <= sid < len(sents)):
            return False
        sent = sents[sid]
        if not isinstance(sent, str):
            return False
        text = f"{t}: {sent}"
        ids = token_ids(text)
        L = len(ids)
        add_cost = L + (newline_cost if len(selected) > 0 else 0)
        if total + add_cost > budget_tokens:
            return False
        key = (t, sid)
        if key in selected_keys:
            return True
        selected_keys.add(key)
        selected.append((t, sid, L, is_evidence))
        total += add_cost
        return True

    for t, sid in sf_pairs:
        if not add_sent(t, sid, True):
            return None, "supporting_over_budget"

    sf_titles = {t for t, _ in sf_pairs}
    candidates = []
    for t, sents in title2sents.items():
        for sid, sent in enumerate(sents):
            key = (t, sid)
            if key in selected_keys or not isinstance(sent, str):
                continue
            text = f"{t}: {sent}"
            L = len(token_ids(text))
            candidates.append((t, sid, L))
    if prefer_same_title:
        candidates.sort(key=lambda x: (0 if x[0] in sf_titles else 1, x[2]))
    else:
        candidates.sort(key=lambda x: x[2])

    for t, sid, L in candidates:
        add_cost = L + (newline_cost if len(selected) > 0 else 0)
        if total + add_cost > budget_tokens:
            continue
        selected_keys.add((t, sid))
        selected.append((t, sid, L, False))
        total += add_cost
        if total >= budget_tokens:
            break

    return selected, "ok"


def main():
    print(f"Loading hotpot_qa distractor train split (n={N_SAMPLES})...")
    ds = load_dataset("hotpot_qa", "distractor", split=f"train[:{N_SAMPLES}]",
                       cache_dir="/cl/work5/hongyu-s/huggingfac/datasets")

    # Match our actual training cfg: hotpot_question_position="later"
    ctx_part = "Context:\n"
    q_part_tmpl = "\nQuestion: {}\n"
    ans_prompt = "Answer:"

    prefix_ids_len = len(token_ids(ctx_part, add_special=True))

    stats = []
    discards = 0
    for ex in ds:
        q_part = q_part_tmpl.format(ex["question"])
        q_ids = token_ids(q_part)
        ans_prompt_ids = token_ids(ans_prompt)
        suffix_len = len(q_ids) + len(ans_prompt_ids)
        ans_ids_len = len(token_ids(" " + ex["answer"].strip()))

        budget_context = BLOCK_SIZE - prefix_ids_len - suffix_len - ans_ids_len - 1
        if budget_context <= 0:
            discards += 1
            continue

        selected, status = build_context_budgeted(ex, budget_context)
        if selected is None:
            discards += 1
            continue

        # cumulative token offsets within the assembled context block
        n_sent = len(selected)
        offsets = []
        cum = 0
        for i, (t, sid, L, is_ev) in enumerate(selected):
            if i > 0:
                cum += 1  # newline
            offsets.append(cum)
            cum += L
        total_ctx_tokens = cum

        ev_token_starts = [offsets[i] for i, (t, sid, L, is_ev) in enumerate(selected) if is_ev]
        ev_sent_idx = [i for i, (t, sid, L, is_ev) in enumerate(selected) if is_ev]

        if not ev_token_starts or total_ctx_tokens == 0:
            continue

        first_ev_token_frac = min(ev_token_starts) / max(1, total_ctx_tokens)
        last_ev_token_frac = max(ev_token_starts) / max(1, total_ctx_tokens)
        first_ev_sent_frac = min(ev_sent_idx) / max(1, n_sent - 1) if n_sent > 1 else 0.0

        stats.append({
            "n_sent": n_sent,
            "total_ctx_tokens": total_ctx_tokens,
            "first_ev_token_frac": first_ev_token_frac,
            "last_ev_token_frac": last_ev_token_frac,
            "first_ev_sent_frac": first_ev_sent_frac,
            "first_ev_sent_idx": min(ev_sent_idx),
        })

    n = len(stats)
    print(f"\nUsable examples: {n} (discarded: {discards})")
    if n == 0:
        return

    import statistics as st
    print(f"budget_context per example: {budget_context} tokens (block_size={BLOCK_SIZE}, fixed prefix/suffix/answer overhead)")
    print(f"\nmean n_sentences in assembled context: {st.mean(s['n_sent'] for s in stats):.1f}")
    print(f"mean total_ctx_tokens used: {st.mean(s['total_ctx_tokens'] for s in stats):.1f} / budget {budget_context}")
    print(f"\nFIRST evidence sentence token position, as fraction of assembled context:")
    fracs = [s["first_ev_token_frac"] for s in stats]
    print(f"  mean={st.mean(fracs):.4f}  median={st.median(fracs):.4f}  p90={sorted(fracs)[int(0.9*len(fracs))]:.4f}  max={max(fracs):.4f}")
    print(f"\nLAST evidence sentence token position, as fraction of assembled context:")
    fracs2 = [s["last_ev_token_frac"] for s in stats]
    print(f"  mean={st.mean(fracs2):.4f}  median={st.median(fracs2):.4f}  p90={sorted(fracs2)[int(0.9*len(fracs2))]:.4f}  max={max(fracs2):.4f}")
    print(f"\nFIRST evidence SENTENCE INDEX (0-based) in assembled context:")
    idxs = [s["first_ev_sent_idx"] for s in stats]
    print(f"  mean={st.mean(idxs):.2f}  median={st.median(idxs)}  max={max(idxs)}  (0 = very first sentence)")
    frac_at_zero = sum(1 for i in idxs if i == 0) / n
    print(f"  fraction of examples where evidence's first sentence is literally sentence #0: {frac_at_zero:.3f}")


if __name__ == "__main__":
    main()

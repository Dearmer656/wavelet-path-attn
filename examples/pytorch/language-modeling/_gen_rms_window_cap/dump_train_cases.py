#!/usr/bin/env python3
"""Dump a few REAL training examples exactly as assembled by run_clm.py's
build_hotpot_qa/build_context_budgeted (dataset_name='mix', hotpot_qa branch,
block_size=512, hotpot_question_position='after'/default -> 'later'),
using the raw HF-hub hotpot_qa distractor train split (the actual source
training reads, confirmed no --hotpot_long_jsonl passed to any train script).

Prints the FULL text that would be tokenized and fed to the model, with each
context sentence tagged [EVIDENCE] or [filler], so the ordering can be
inspected directly rather than only via aggregate stats.
"""
from transformers import GPT2Tokenizer
from datasets import load_dataset

BLOCK_SIZE = 512
N_CASES = 6

tok = GPT2Tokenizer.from_pretrained("gpt2")


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
    """Faithful reimplementation of run_clm.py:3237."""
    title2sents = build_title2sents(ex)
    newline_ids = token_ids("\n")
    newline_cost = len(newline_ids)
    sf_pairs = get_supporting_pairs(ex)
    if not sf_pairs:
        return None

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
        selected.append((t, sid, text, is_evidence))
        total += add_cost
        return True

    for t, sid in sf_pairs:
        if not add_sent(t, sid, True):
            return None

    sf_titles = {t for t, _ in sf_pairs}
    candidates = []
    for t, sents in title2sents.items():
        for sid, sent in enumerate(sents):
            key = (t, sid)
            if key in selected_keys or not isinstance(sent, str):
                continue
            text = f"{t}: {sent}"
            L = len(token_ids(text))
            candidates.append((t, sid, text, L))
    if prefer_same_title:
        candidates.sort(key=lambda x: (0 if x[0] in sf_titles else 1, x[3]))
    else:
        candidates.sort(key=lambda x: x[3])

    for t, sid, text, L in candidates:
        add_cost = L + (newline_cost if len(selected) > 0 else 0)
        if total + add_cost > budget_tokens:
            continue
        selected_keys.add((t, sid))
        selected.append((t, sid, text, False))
        total += add_cost
        if total >= budget_tokens:
            break

    return selected


def main():
    ds = load_dataset("hotpot_qa", "distractor", split="train",
                       cache_dir="/cl/work5/hongyu-s/huggingfac/datasets")

    ctx_part = "Context:\n"
    q_part_tmpl = "\nQuestion: {}\n"
    ans_prompt = "Answer:"
    prefix_ids_len = len(token_ids(ctx_part, add_special=True))

    shown = 0
    idx = 0
    while shown < N_CASES and idx < len(ds):
        ex = ds[idx]
        idx += 1
        q_part = q_part_tmpl.format(ex["question"])
        q_ids = token_ids(q_part)
        ans_prompt_ids = token_ids(ans_prompt)
        suffix_len = len(q_ids) + len(ans_prompt_ids)
        ans_ids_len = len(token_ids(" " + ex["answer"].strip()))
        budget_context = BLOCK_SIZE - prefix_ids_len - suffix_len - ans_ids_len - 1
        if budget_context <= 0:
            continue
        selected = build_context_budgeted(ex, budget_context)
        if selected is None:
            continue

        shown += 1
        print("=" * 100)
        print(f"CASE {shown}  (hf train idx={idx-1}, id={ex['id']})")
        print(f"question: {ex['question']}")
        print(f"answer:   {ex['answer']}")
        print(f"n_supporting_facts: {len(get_supporting_pairs(ex))}   n_context_sents_selected: {len(selected)}")
        print("-" * 100)
        print("### FULL PROMPT TEXT AS FED TO MODEL (training input_ids source) ###")
        print(ctx_part, end="")
        for i, (t, sid, text, is_ev) in enumerate(selected):
            tag = "[EVIDENCE]" if is_ev else "[filler]  "
            prefix_nl = "\n" if i > 0 else ""
            print(f"{prefix_nl}{tag} {text}")
        print(q_part_tmpl.format(ex["question"]), end="")
        print(ans_prompt, ex["answer"])
        print()
        ev_positions = [i for i, (t, sid, text, is_ev) in enumerate(selected) if is_ev]
        print(f"--> evidence sentence indices within assembled context: {ev_positions} / total {len(selected)} sentences")
        print(f"--> evidence occupies sentence-index range [0, {max(ev_positions)}] out of [0, {len(selected)-1}]")


if __name__ == "__main__":
    main()

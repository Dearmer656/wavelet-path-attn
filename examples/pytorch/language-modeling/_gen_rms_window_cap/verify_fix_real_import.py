#!/usr/bin/env python3
"""Import build_context_budgeted DIRECTLY from the real run_clm.py (post-fix)
and rerun the exact same statistics + case dump against real training data,
to verify the fix against the actual production code (not a reimplementation).
"""
import sys
import statistics as st

sys.path.insert(0, "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling")

import run_clm  # noqa: E402
from transformers import GPT2Tokenizer  # noqa: E402
from datasets import load_dataset  # noqa: E402

BLOCK_SIZE = 512
N_STATS = 500
N_DUMP = 6

tok = GPT2Tokenizer.from_pretrained("gpt2")
run_clm.GLOBAL_TOKENIZER = tok


def get_supporting_pairs(ex):
    return run_clm._get_supporting_pairs(ex)


def main():
    ds = load_dataset("hotpot_qa", "distractor", split=f"train[:{N_STATS}]",
                       cache_dir="/cl/work5/hongyu-s/huggingfac/datasets")

    ctx_part = "Context:\n"
    q_part_tmpl = "\nQuestion: {}\n"
    ans_prompt = "Answer:"

    def token_ids(text, add_special=False):
        return tok(text, add_special_tokens=add_special, truncation=False)["input_ids"]

    prefix_ids_len = len(token_ids(ctx_part, add_special=True))

    stats = []
    dumped = 0
    for i, ex in enumerate(ds):
        q_part = q_part_tmpl.format(ex["question"])
        q_ids = token_ids(q_part)
        ans_prompt_ids = token_ids(ans_prompt)
        suffix_len = len(q_ids) + len(ans_prompt_ids)
        ans_ids_len = len(token_ids(" " + ex["answer"].strip()))
        budget_context = BLOCK_SIZE - prefix_ids_len - suffix_len - ans_ids_len - 1
        if budget_context <= 0:
            continue

        context_text, context_ids, status = run_clm.build_context_budgeted(
            ex, budget_context, prefer_same_title=True, min_tokens=0
        )
        if context_text is None:
            continue

        sf_pairs = get_supporting_pairs(ex)
        ev_keys = set(sf_pairs)
        lines = context_text.split("\n")
        # Recompute which line indices correspond to evidence by re-deriving (title, approx) —
        # simplest robust way: rebuild title2sents and match "title: sentence" text against sf sentences.
        title2sents = run_clm._build_title2sents(ex)
        ev_texts = set()
        for t, sid in sf_pairs:
            sents = title2sents.get(t, [])
            if 0 <= sid < len(sents) and isinstance(sents[sid], str):
                ev_texts.add(f"{t}: {sents[sid]}")

        ev_idx = [j for j, line in enumerate(lines) if line in ev_texts]
        n_lines = len(lines)
        if not ev_idx or n_lines == 0:
            continue

        stats.append({
            "n_lines": n_lines,
            "ev_idx": ev_idx,
            "first_ev_frac": min(ev_idx) / max(1, n_lines - 1) if n_lines > 1 else 0.0,
        })

        if dumped < N_DUMP:
            dumped += 1
            print("=" * 100)
            print(f"CASE {dumped} (hf idx={i}, id={ex['id']})  status={status}")
            print(f"question: {ex['question']}   answer: {ex['answer']}")
            print("-" * 100)
            print(ctx_part, end="")
            for j, line in enumerate(lines):
                tag = "[EVIDENCE]" if line in ev_texts else "[filler]  "
                sep = "\n" if j > 0 else ""
                print(f"{sep}{tag} {line}")
            print(f"\n--> evidence line indices: {ev_idx} / total {n_lines} lines")

    n = len(stats)
    print("\n" + "#" * 100)
    print(f"AGGREGATE STATS over {n} real training examples (post-fix, via REAL run_clm.build_context_budgeted import)")
    fracs = [s["first_ev_frac"] for s in stats]
    first_idx = [min(s["ev_idx"]) for s in stats]
    print(f"first evidence line index: mean={st.mean(first_idx):.2f} median={st.median(first_idx)} max={max(first_idx)}")
    print(f"first evidence position as fraction of context: mean={st.mean(fracs):.4f} median={st.median(fracs):.4f}")
    frac_at_zero = sum(1 for i in first_idx if i == 0) / n
    print(f"fraction of examples where evidence's first line is still literally line #0: {frac_at_zero:.3f}")


if __name__ == "__main__":
    main()

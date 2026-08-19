#!/usr/bin/env python3
"""Build several REAL training batches (via the real run_clm.build_context_budgeted,
post-fix) and check whether evidence position within the assembled context is
actually uniformly distributed now, not just "less front-loaded on average".

Reports a decile histogram of each example's evidence-sentence positions
(every evidence line's fractional position within the assembled context,
not just the first one) plus a chi-square goodness-of-fit test against uniform.
"""
import sys
import statistics as st

sys.path.insert(0, "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling")

import run_clm  # noqa: E402
from transformers import GPT2Tokenizer  # noqa: E402
from datasets import load_dataset  # noqa: E402

BLOCK_SIZE = 512
BATCH_SIZE = 64
N_BATCHES = 4  # 4 real training batches worth = 256 examples

tok = GPT2Tokenizer.from_pretrained("gpt2")
run_clm.GLOBAL_TOKENIZER = tok


def token_ids(text, add_special=False):
    return tok(text, add_special_tokens=add_special, truncation=False)["input_ids"]


def main():
    n_total = BATCH_SIZE * N_BATCHES
    ds = load_dataset("hotpot_qa", "distractor", split=f"train[:{n_total*2}]",
                       cache_dir="/cl/work5/hongyu-s/huggingfac/datasets")

    ctx_part = "Context:\n"
    q_part_tmpl = "\nQuestion: {}\n"
    ans_prompt = "Answer:"
    prefix_ids_len = len(token_ids(ctx_part, add_special=True))

    all_ev_fracs = []       # every evidence line's fractional position (0..1), across ALL examples
    first_ev_fracs = []     # only the first evidence line's fractional position per example
    n_kept = 0
    batch_num = 0
    idx = 0

    while n_kept < n_total and idx < len(ds):
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

        context_text, context_ids, status = run_clm.build_context_budgeted(
            ex, budget_context, prefer_same_title=True, min_tokens=0
        )
        if context_text is None:
            continue

        sf_pairs = run_clm._get_supporting_pairs(ex)
        title2sents = run_clm._build_title2sents(ex)
        ev_texts = set()
        for t, sid in sf_pairs:
            sents = title2sents.get(t, [])
            if 0 <= sid < len(sents) and isinstance(sents[sid], str):
                ev_texts.add(f"{t}: {sents[sid]}")

        lines = context_text.split("\n")
        n_lines = len(lines)
        ev_idx = [j for j, line in enumerate(lines) if line in ev_texts]
        if not ev_idx or n_lines <= 1:
            continue

        for j in ev_idx:
            all_ev_fracs.append(j / (n_lines - 1))
        first_ev_fracs.append(min(ev_idx) / (n_lines - 1))
        n_kept += 1
        if n_kept % BATCH_SIZE == 0:
            batch_num += 1

    print(f"Collected {n_kept} usable examples across {batch_num} real training-size batches (batch_size={BATCH_SIZE})")
    print(f"Total evidence-line position samples: {len(all_ev_fracs)}\n")

    # decile histogram of ALL evidence-line positions
    n_bins = 10
    bins = [0] * n_bins
    for f in all_ev_fracs:
        b = min(n_bins - 1, int(f * n_bins))
        bins[b] += 1
    total = len(all_ev_fracs)
    expected = total / n_bins
    print("Decile histogram of EVERY evidence-sentence position within its assembled context:")
    print(f"{'bin':>12} {'count':>7} {'frac':>7} {'expected(uniform)':>18}")
    chi2 = 0.0
    for i, c in enumerate(bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        print(f"[{lo:.1f},{hi:.1f}) {c:>7} {c/total:>7.3f} {expected:>18.1f}")
        chi2 += (c - expected) ** 2 / expected
    print(f"\nchi-square statistic (df={n_bins-1}) vs uniform: {chi2:.2f}")
    # rough critical value for df=9 at alpha=0.05 is 16.92; just report, don't need scipy
    print("(critical value at alpha=0.05, df=9 is ~16.92; below that = consistent with uniform)")

    print("\n--- first-evidence-only decile histogram (matches earlier stat definition) ---")
    bins2 = [0] * n_bins
    for f in first_ev_fracs:
        b = min(n_bins - 1, int(f * n_bins))
        bins2[b] += 1
    total2 = len(first_ev_fracs)
    for i, c in enumerate(bins2):
        lo, hi = i / n_bins, (i + 1) / n_bins
        print(f"[{lo:.1f},{hi:.1f}) {c:>7} {c/total2:>7.3f}")

    print(f"\nfirst-evidence position: mean={st.mean(first_ev_fracs):.4f} median={st.median(first_ev_fracs):.4f} stdev={st.stdev(first_ev_fracs):.4f}")
    frac_at_zero = sum(1 for f in first_ev_fracs if f == 0.0) / total2
    print(f"fraction still exactly at position 0: {frac_at_zero:.3f}  (uniform-over-index expectation ~ 1/mean_n_lines, i.e. small but nonzero)")


if __name__ == "__main__":
    main()

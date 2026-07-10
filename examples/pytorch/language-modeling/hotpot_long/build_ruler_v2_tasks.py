#!/usr/bin/env python3
"""
Build RULER-v2 task splits with controlled needle DEPTH and diverse task types.

Motivation (PAT-222): the slash+sink motif is a positional prior. Its benefit should
differ by task type — point retrieval (NIAH) relies on long-range positional lookup,
aggregation needs uniform global coverage, QA is mixed. Depth control lets us build
the classic length x depth heatmap (baseline vs motif).

Tasks (all short answers, scored by word-containment):
  niah.single       one magic-number needle at depth d
  niah.multikey     gold needle + 4 distractor needles at other depths
  niah.passkey      classic passkey retrieval at depth d
  agg.freq_word     target special word appears 8x scattered, distractors 2x; ask most frequent
  qa.hotpot         2 gold HotpotQA paragraphs inserted at depth d, real question

Output rows: {input, outputs, ruler_config, length, depth}
  - length: target LLaMA token count of the full prompt
  - depth:  fractional position of the (gold) needle in the context body

Token counting uses the LLaMA-2 tokenizer directly (exact control for the eval model).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

from transformers import AutoTokenizer

FILLER_SENTENCES = [
    "The grass is green.",
    "The sky is blue.",
    "The sun is yellow.",
    "Here we go.",
    "There and back again.",
    "The wind is calm.",
    "The river flows.",
    "The stars are bright.",
    "The moon is full.",
    "The road is long.",
    "The path is clear.",
    "The lake is still.",
    "The hill is steep.",
    "The field is wide.",
]

WORD_POOL = [
    "apple", "island", "guitar", "meadow", "candle", "harbor", "walnut", "breeze",
    "copper", "lantern", "orchid", "pebble", "saddle", "timber", "velvet", "willow",
]

KEY_POOL = [
    "aurora", "basalt", "cinder", "dune", "ember", "fjord", "glacier", "harvest",
    "iris", "juniper", "krypton", "lagoon", "mesa", "nectar", "onyx", "prairie",
]

DEPTHS = [0.1, 0.3, 0.5, 0.7, 0.9]


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


class FillerBuilder:
    """Assemble filler bodies to a token budget using precomputed sentence lengths."""

    def __init__(self, tok):
        self.tok = tok
        # token length of each sentence when concatenated with a leading space
        self.sent_toks = [len(tok(" " + s, add_special_tokens=False)["input_ids"])
                          for s in FILLER_SENTENCES]

    def count(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=True)["input_ids"])

    def body(self, rng: random.Random, budget_tokens: int) -> List[str]:
        """Return list of filler sentences totalling ~budget_tokens."""
        out, total = [], 0
        while total < budget_tokens:
            i = rng.randrange(len(FILLER_SENTENCES))
            if total + self.sent_toks[i] > budget_tokens + 4:
                break
            out.append(FILLER_SENTENCES[i])
            total += self.sent_toks[i]
        return out


def assemble(fb: FillerBuilder, rng: random.Random, header: str, qtail: str,
             target_tokens: int, insertions: List[tuple]) -> str:
    """Build header + filler-with-insertions + qtail at ~target_tokens.

    insertions: list of (depth_fraction, text) placed at that fraction of the body.
    Iteratively adjusts the filler budget so the final LLaMA token count lands within
    [target-8, target].
    """
    fixed = header + qtail + "".join(t for _, t in insertions)
    budget = target_tokens - fb.count(fixed) - 8
    for _ in range(6):
        sents = fb.body(rng, budget)
        n = max(len(sents), 1)
        slots = sorted(insertions, key=lambda x: x[0])
        parts, cursor = [], 0
        for frac, text in slots:
            k = min(int(round(frac * n)), n)
            parts.append(" ".join(sents[cursor:k]))
            parts.append(text)
            cursor = k
        parts.append(" ".join(sents[cursor:]))
        prompt = header + " ".join(p for p in parts if p) + qtail
        cnt = fb.count(prompt)
        if target_tokens - 8 <= cnt <= target_tokens:
            return prompt
        budget += (target_tokens - 4) - cnt
    return prompt  # last attempt, close enough


# ── task generators ──────────────────────────────────────────────────────────

def gen_niah_single(fb, rng, target, depth):
    key = rng.choice(KEY_POOL)
    val = str(rng.randint(100000, 999999))
    needle = f" The special magic number for {key} is {val}. "
    header = "Read the following text carefully. A special magic number is hidden inside. "
    qtail = (f"\nQuestion: What is the special magic number for {key}?"
             "\nAnswer with digits only.")
    prompt = assemble(fb, rng, header, qtail, target, [(depth, needle)])
    return prompt, [val], "niah.single"


def gen_niah_multikey(fb, rng, target, depth):
    keys = rng.sample(KEY_POOL, 5)
    vals = [str(rng.randint(100000, 999999)) for _ in range(5)]
    gold_key, gold_val = keys[0], vals[0]
    ins = [(depth, f" The special magic number for {gold_key} is {gold_val}. ")]
    for k, v in zip(keys[1:], vals[1:]):
        d = rng.choice([x for x in DEPTHS if abs(x - depth) > 0.05])
        ins.append((d + rng.uniform(-0.04, 0.04), f" The special magic number for {k} is {v}. "))
    header = "Read the following text carefully. Several special magic numbers are hidden inside. "
    qtail = (f"\nQuestion: What is the special magic number for {gold_key}?"
             "\nAnswer with digits only.")
    prompt = assemble(fb, rng, header, qtail, target, ins)
    return prompt, [gold_val], "niah.multikey"


def gen_niah_passkey(fb, rng, target, depth):
    val = str(rng.randint(10000, 99999))
    needle = f" The pass key is {val}. Remember it. {val} is the pass key. "
    header = "There is a pass key hidden in the following text. Find it and remember it. "
    qtail = "\nQuestion: What is the pass key?\nAnswer with digits only."
    prompt = assemble(fb, rng, header, qtail, target, [(depth, needle)])
    return prompt, [val], "niah.passkey"


def gen_agg_freq_word(fb, rng, target, depth):
    words = rng.sample(WORD_POOL, 5)
    gold = words[0]
    ins = []
    for d in [0.08, 0.2, 0.32, 0.44, 0.56, 0.68, 0.8, 0.92]:
        ins.append((d + rng.uniform(-0.03, 0.03), f" Special word: {gold}. "))
    for w in words[1:]:
        for _ in range(2):
            ins.append((rng.uniform(0.05, 0.95), f" Special word: {w}. "))
    header = ("The following text contains lines of the form 'Special word: X.'. "
              "Count how often each special word appears. ")
    qtail = ("\nQuestion: Which special word appears most frequently?"
             "\nAnswer with one word.")
    prompt = assemble(fb, rng, header, qtail, target, ins)
    # depth is not meaningful for aggregation (needles span the body); record 0.5
    return prompt, [gold], "agg.freq_word"


def load_hotpot_cases(path: Path, n_needed: int) -> List[Dict]:
    out = []
    seen = set()
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            if r["base_id"] in seen:
                continue
            ans = r["answer"].strip()
            if not ans or len(ans.split()) > 3:
                continue
            # collect gold paragraphs (title + sentences) via supporting_facts titles
            gold_titles = {t for t, _ in r["supporting_facts"]}
            paras = []
            for title, sents in r["context"]:
                if title in gold_titles:
                    paras.append(f"{title}: " + " ".join(sents))
            if len(paras) < 2:
                continue
            seen.add(r["base_id"])
            out.append({"question": r["question"], "answers": [ans] + list(r.get("answer_aliases", [])),
                        "paras": paras[:2]})
            if len(out) >= n_needed:
                break
    return out


def gen_qa_hotpot(fb, rng, target, depth, hp_case):
    needle = " " + " ".join(hp_case["paras"]) + " "
    header = ("Read the following text carefully. It contains passages that answer a question "
              "asked at the end. ")
    qtail = (f"\nQuestion: {hp_case['question']}"
             "\nAnswer with a short phrase.")
    prompt = assemble(fb, rng, header, qtail, target, [(depth, needle)])
    return prompt, hp_case["answers"], "qa.hotpot"


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_tok", default="NousResearch/Llama-2-7b-chat-hf")
    ap.add_argument("--cache_dir", default="/cl/work5/hongyu-s/huggingfac")
    ap.add_argument("--hotpot_jsonl", type=Path, default=Path(
        "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/hotpot_long_dev_uniform.jsonl"))
    ap.add_argument("--outdir", type=Path, default=Path(
        "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data"))
    ap.add_argument("--lengths", type=int, nargs="+",
                    default=[2048, 4080, 4608, 5120, 6144, 8192])
    ap.add_argument("--per_depth", type=int, default=5,
                    help="cases per (task, length, depth); 5 depths -> 25 per task-length")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_tok, cache_dir=args.cache_dir)
    fb = FillerBuilder(tok)

    n_hp = len(args.lengths) * len(DEPTHS) * args.per_depth
    hp_cases = load_hotpot_cases(args.hotpot_jsonl, n_hp)
    print(f"[v2] hotpot cases loaded: {len(hp_cases)}", flush=True)

    for L in args.lengths:
        rng = random.Random(1000 + L)
        rows = []
        hp_iter = iter(hp_cases)
        for depth in DEPTHS:
            for i in range(args.per_depth):
                for gen in (gen_niah_single, gen_niah_multikey, gen_niah_passkey):
                    p, golds, cfg = gen(fb, rng, L, depth)
                    rows.append({"input": p, "outputs": golds, "ruler_config": cfg,
                                 "length": L, "depth": depth})
                p, golds, cfg = gen_agg_freq_word(fb, rng, L, depth)
                rows.append({"input": p, "outputs": golds, "ruler_config": cfg,
                             "length": L, "depth": 0.5})
                try:
                    hp = next(hp_iter)
                except StopIteration:
                    hp_iter = iter(hp_cases); hp = next(hp_iter)
                p, golds, cfg = gen_qa_hotpot(fb, rng, L, depth, hp)
                rows.append({"input": p, "outputs": golds, "ruler_config": cfg,
                             "length": L, "depth": depth})
        rng.shuffle(rows)
        out = args.outdir / f"ruler_v2_{L}.jsonl"
        write_jsonl(out, rows)
        lens = [fb.count(r["input"]) for r in rows[:20]]
        print(f"[v2] L={L}: n={len(rows)} sample-token-range=[{min(lens)},{max(lens)}] -> {out}",
              flush=True)


if __name__ == "__main__":
    main()

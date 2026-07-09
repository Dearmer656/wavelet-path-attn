#!/usr/bin/env python3
"""
Build RULER train/eval splits with controlled prompt token lengths.

Input:
  - hotpot_long/data/ruler_smoke_100.jsonl (task templates + gold answers)

Output:
  - hotpot_long/data/ruler_train_512.jsonl
  - hotpot_long/data/ruler_eval_2048.jsonl
  - hotpot_long/data/ruler_eval_4096.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from tokenizers import Tokenizer


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def count_tokens(tok: Tokenizer, text: str) -> int:
    return len(tok.encode(text).ids)


def split_prompt(prompt: str) -> Tuple[str, str]:
    marker = r"\nQuestion:"
    idx = prompt.find(marker)
    if idx < 0:
        # Fallback: keep question as-is at the end.
        return prompt, ""
    return prompt[:idx], prompt[idx:]


# RULER-style neutral filler sentences (no overlap with task vocabulary).
# Reference: github.com/NVIDIA/RULER — "The grass is green..." repeated pattern.
_FILLER_SENTENCES = [
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
]


def make_noise_line(rng: random.Random, j: int) -> str:
    # Pick 2-3 neutral sentences; no task-relevant keywords (no tag/flag/code/key/value).
    n = rng.randint(2, 3)
    chosen = [rng.choice(_FILLER_SENTENCES) for _ in range(n)]
    return r"\n" + " ".join(chosen)


def stretch_prompt(
    tok: Tokenizer,
    prompt: str,
    target_tokens: int,
    rng: random.Random,
    max_overflow: int = 0,
) -> str:
    prefix, qtail = split_prompt(prompt)
    lines: List[str] = []
    j = 0

    cur = prefix + qtail
    cur_len = count_tokens(tok, cur)
    if cur_len >= target_tokens:
        return cur

    # Grow with medium-size neutral filler lines.
    while cur_len < target_tokens:
        cand_lines = lines + [make_noise_line(rng, j)]
        cand = prefix + "".join(cand_lines) + qtail
        cand_len = count_tokens(tok, cand)
        if cand_len <= target_tokens + max_overflow:
            lines = cand_lines
            cur = cand
            cur_len = cand_len
            j += 1
        else:
            break

    # Fine-grained padding — insert BEFORE qtail so question stays at end.
    fine_sents = ["The path is clear.", "The lake is still.", "The hill is steep.", "The field is wide."]
    inner = prefix + "".join(lines)   # everything before qtail
    while cur_len < target_tokens:
        w = rng.choice(fine_sents)
        cand_inner = inner + r"\n" + w
        cand = cand_inner + qtail
        cand_len = count_tokens(tok, cand)
        if cand_len <= target_tokens + max_overflow:
            inner = cand_inner
            cur = cand
            cur_len = cand_len
        else:
            break

    return cur


def build_split(
    base_rows: List[Dict],
    tok: Tokenizer,
    out_len_label: int,
    target_tokens: int,
    out_count: int,
    seed: int,
) -> List[Dict]:
    rng = random.Random(seed)
    out: List[Dict] = []
    for i in range(out_count):
        src = dict(base_rows[i % len(base_rows)])
        prompt = src["input"]
        stretched = stretch_prompt(tok, prompt, target_tokens, rng)
        row = {
            "input": stretched,
            "outputs": src["outputs"],
            "ruler_config": src.get("ruler_config", "unknown"),
            "length": out_len_label,
        }
        out.append(row)
    return out


def summarize(name: str, rows: List[Dict], tok: Tokenizer) -> None:
    lens = [count_tokens(tok, r["input"]) for r in rows]
    lens_sorted = sorted(lens)
    mid = lens_sorted[len(lens_sorted) // 2]
    p90 = lens_sorted[int(len(lens_sorted) * 0.9)]
    print(
        f"{name}: n={len(rows)} min={min(lens)} p50={mid} p90={p90} max={max(lens)} "
        f"labels={sorted(set(r['length'] for r in rows))}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--infile",
        type=Path,
        default=Path(
            "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/ruler_smoke_100.jsonl"
        ),
    )
    ap.add_argument(
        "--tokenizer_json",
        type=Path,
        default=Path(
            "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/ruler_ft_matrix_20260525/path_pa_s42_save1k/tokenizer.json"
        ),
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=Path("/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data"),
    )
    ap.add_argument("--train_count", type=int, default=2000)
    ap.add_argument("--eval_count", type=int, default=200)
    args = ap.parse_args()

    base_rows = load_jsonl(args.infile)
    tok = Tokenizer.from_file(str(args.tokenizer_json))

    train_rows = build_split(base_rows, tok, out_len_label=512, target_tokens=512, out_count=args.train_count, seed=42)
    eval_2048_rows = build_split(
        base_rows, tok, out_len_label=2048, target_tokens=2048, out_count=args.eval_count, seed=43
    )
    eval_4096_rows = build_split(
        base_rows, tok, out_len_label=4096, target_tokens=4096, out_count=args.eval_count, seed=44
    )
    eval_4080_rows = build_split(
        base_rows, tok, out_len_label=4080, target_tokens=4080, out_count=args.eval_count, seed=46
    )
    eval_8192_rows = build_split(
        base_rows, tok, out_len_label=8192, target_tokens=8192, out_count=args.eval_count, seed=45
    )

    train_path = args.outdir / "ruler_train_512.jsonl"
    eval_2048_path = args.outdir / "ruler_eval_2048.jsonl"
    eval_4080_path = args.outdir / "ruler_eval_4080.jsonl"
    eval_4096_path = args.outdir / "ruler_eval_4096.jsonl"
    eval_8192_path = args.outdir / "ruler_eval_8192.jsonl"

    write_jsonl(train_path, train_rows)
    write_jsonl(eval_2048_path, eval_2048_rows)
    write_jsonl(eval_4080_path, eval_4080_rows)
    write_jsonl(eval_4096_path, eval_4096_rows)
    write_jsonl(eval_8192_path, eval_8192_rows)

    summarize("train_512", train_rows, tok)
    summarize("eval_2048", eval_2048_rows, tok)
    summarize("eval_4080", eval_4080_rows, tok)
    summarize("eval_4096", eval_4096_rows, tok)
    summarize("eval_8192", eval_8192_rows, tok)
    print(f"wrote: {train_path}")
    print(f"wrote: {eval_2048_path}")
    print(f"wrote: {eval_4080_path}")
    print(f"wrote: {eval_4096_path}")
    print(f"wrote: {eval_8192_path}")


if __name__ == "__main__":
    main()

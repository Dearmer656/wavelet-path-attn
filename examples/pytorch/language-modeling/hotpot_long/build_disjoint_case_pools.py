#!/usr/bin/env python3
"""Build mutually disjoint Hotpot case pools for motif-retention analysis."""

import argparse
import json
from collections import OrderedDict
from pathlib import Path


POOL_SPECS = [
    ("train_dict_pool", 50),
    ("oracle_dict_pool_2048", 50),
    ("oracle_dict_pool_4096", 50),
    ("eval_pool", 50),
]


def pair_cases_by_base_id(jsonl_path: Path):
    by_base = OrderedDict()
    by_len = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            base_id = rec["base_id"]
            target_len = rec["meta"]["target_total_tokens"]
            by_base.setdefault(base_id, {})
            by_base[base_id][target_len] = rec
            by_len.setdefault(base_id, []).append(target_len)
    eligible = []
    for base_id, records in by_base.items():
        if all(L in records for L in (512, 2048, 4096)):
            eligible.append((base_id, records))
    return eligible


def write_pool(path: Path, selected):
    with open(path, "w", encoding="utf-8") as f:
        for _, records in selected:
            for length in (512, 2048, 4096):
                f.write(json.dumps(records[length], ensure_ascii=False) + "\n")


def load_base_ids(path: Path):
    base_ids = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            base_ids.append(json.loads(line)["base_id"])
    return base_ids


def run(args):
    root = Path(__file__).resolve().parent
    jsonl_path = args.jsonl or (root / "data" / "hotpot_long_dev_uniform.jsonl")
    out_dir = args.out_dir or (root / "data" / "motif_retention_pools")
    out_dir.mkdir(parents=True, exist_ok=True)

    eligible = pair_cases_by_base_id(jsonl_path)
    need = sum(n for _, n in POOL_SPECS)
    if len(eligible) < need:
        raise SystemExit(
            f"Need {need} eligible base_ids with lengths 512/2048/4096, but only found {len(eligible)} in {jsonl_path}"
        )

    offsets = 0
    written = {}
    for pool_name, size in POOL_SPECS:
        selected = eligible[offsets:offsets + size]
        if len(selected) != size:
            raise SystemExit(f"Internal error selecting {pool_name}: expected {size}, got {len(selected)}")
        offsets += size
        out_path = out_dir / f"{pool_name}.jsonl"
        write_pool(out_path, selected)
        written[pool_name] = out_path
        print(f"{pool_name}: {len(selected)} base_ids -> {out_path}")

    seen = {}
    for pool_name, path in written.items():
        base_ids = load_base_ids(path)
        uniq = sorted(set(base_ids))
        print(f"{pool_name}: {len(uniq)} base_ids")
        for base_id in uniq:
            if base_id in seen:
                raise SystemExit(
                    f"base_id {base_id} appears in both {seen[base_id]} and {pool_name}; pools must be disjoint"
                )
            seen[base_id] = pool_name

    print("All four pools are disjoint by base_id.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

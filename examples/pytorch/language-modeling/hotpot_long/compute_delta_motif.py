#!/usr/bin/env python3
"""Combine motif-retention summaries into Delta_motif tables."""

import argparse
import json
from pathlib import Path


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwab_summary", required=True)
    parser.add_argument("--paonly_summary", required=True)
    parser.add_argument("--out_dir", type=Path, default=root / "analysis_outputs" / "motif_retention_gap")
    return parser.parse_args()


def load_summary(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    qwab = load_summary(Path(args.qwab_summary))
    pa = load_summary(Path(args.paonly_summary))

    lines = [
        "# Delta Motif Summary",
        "",
        "| Length | G_QWAB | G_PAonly | Delta_motif |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in ("2048", "4096"):
        g_q = float(qwab["lengths"][key]["G_m"])
        g_p = float(pa["lengths"][key]["G_m"])
        delta = g_q - g_p
        lines.append(f"| {key} | {g_q:.6f} | {g_p:.6f} | {delta:.6f} |")
    lines.append("")
    lines.append("Negative Delta_motif means QWAB better preserves the train-length motif space under extrapolation.")

    out_path = out_dir / "delta_motif_summary.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    run(parse_args())

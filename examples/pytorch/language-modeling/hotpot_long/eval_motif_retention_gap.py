#!/usr/bin/env python3
"""Evaluate train-length motif retention under extrapolation."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from dump_fixed_dict_fit_quality import fit_against_dictionary
from run_head_ablation_eval import build_capture_input_ids, capture_ext_logit_maps, find_path_attention_modules, load_path_attn_model, preprocess_head_features


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["PA-only", "QWAB"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--traindict_dir", required=True)
    parser.add_argument("--oracledict_dir_2048", required=True)
    parser.add_argument("--oracledict_dir_4096", required=True)
    parser.add_argument("--eval_jsonl", type=Path, default=root / "data" / "motif_retention_pools" / "eval_pool.jsonl")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--pool_size", type=int, default=128)
    parser.add_argument("--preprocessing", default="salient")
    parser.add_argument("--out_dir", type=Path, default=root / "analysis_outputs" / "motif_retention_gap")
    return parser.parse_args()


def load_layer_dicts(root_dir: Path):
    V_by_layer = {}
    for layer_idx in range(12):
        V_by_layer[layer_idx] = np.load(root_dir / f"layer_{layer_idx:02d}" / "V.npy").astype(np.float64, copy=False)
    return V_by_layer


def load_eval_cases(jsonl_path: Path, target_len: int):
    cases = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["meta"].get("target_total_tokens") == target_len:
                cases.append(rec)
    return cases


def model_tag(model: str) -> str:
    return "qwab" if model == "QWAB" else "paonly"


def run_one_length(args, model, tokenizer, device, path_modules, head_dim_scale, V_train, V_oracle, cases, length):
    rows = []
    case_gap_rows = []
    for rec in cases:
        case_id = rec["base_id"]
        input_ids = build_capture_input_ids(tokenizer, rec, length, device)
        logit_maps = capture_ext_logit_maps(model, path_modules, input_ids, args.model == "QWAB", head_dim_scale)
        head_features, valid_heads = preprocess_head_features(logit_maps, args.pool_size, args.preprocessing)
        gaps = []
        for head in valid_heads:
            x = head_features[head]
            e_train = fit_against_dictionary(x, V_train[head.layer_idx])["recon_error"]
            e_oracle = fit_against_dictionary(x, V_oracle[head.layer_idx])["recon_error"]
            gap = e_train - e_oracle
            gaps.append(gap)
            rows.append({
                "model": args.model,
                "length": length,
                "case_id": case_id,
                "layer": head.layer_idx,
                "head": head.head_idx,
                "e_train": e_train,
                "e_oracle": e_oracle,
                "gap": gap,
            })
        case_gap_rows.append({"case_id": case_id, "gap_mean": float(np.mean(gaps)) if gaps else float("nan")})
    per_case = np.array([r["gap_mean"] for r in case_gap_rows], dtype=np.float64)
    summary = {
        "model": args.model,
        "length": length,
        "n_cases": int(len(case_gap_rows)),
        "G_m": float(np.mean(per_case)) if per_case.size else float("nan"),
        "case_gap_mean": float(np.mean(per_case)) if per_case.size else float("nan"),
        "case_gap_std": float(np.std(per_case, ddof=1)) if per_case.size > 1 else 0.0,
        "case_gap_sem": float(np.std(per_case, ddof=1) / np.sqrt(per_case.size)) if per_case.size > 1 else 0.0,
        "case_gaps": case_gap_rows,
    }
    return rows, summary


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=False)
    model = load_path_attn_model(args.checkpoint, device)
    path_modules = find_path_attention_modules(model)
    head_dim = model.config.n_embd // model.config.n_head
    head_dim_scale = head_dim ** -0.5

    V_train = load_layer_dicts(Path(args.traindict_dir))
    V_oracle_2048 = load_layer_dicts(Path(args.oracledict_dir_2048))
    V_oracle_4096 = load_layer_dicts(Path(args.oracledict_dir_4096))

    eval_cases_2048 = load_eval_cases(args.eval_jsonl, 2048)
    eval_cases_4096 = load_eval_cases(args.eval_jsonl, 4096)

    all_rows = []
    summaries = {}
    rows_2048, summary_2048 = run_one_length(
        args, model, tokenizer, device, path_modules, head_dim_scale, V_train, V_oracle_2048, eval_cases_2048, 2048
    )
    rows_4096, summary_4096 = run_one_length(
        args, model, tokenizer, device, path_modules, head_dim_scale, V_train, V_oracle_4096, eval_cases_4096, 4096
    )
    all_rows.extend(rows_2048)
    all_rows.extend(rows_4096)
    summaries["2048"] = summary_2048
    summaries["4096"] = summary_4096

    rows_path = out_dir / f"motif_retention_rows_{model_tag(args.model)}.csv"
    with open(rows_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "length", "case_id", "layer", "head", "e_train", "e_oracle", "gap"]
        )
        writer.writeheader()
        writer.writerows(all_rows)

    summary = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "traindict_dir": args.traindict_dir,
        "oracledict_dir_2048": args.oracledict_dir_2048,
        "oracledict_dir_4096": args.oracledict_dir_4096,
        "rows_csv": str(rows_path),
        "lengths": summaries,
    }
    with open(out_dir / f"motif_retention_summary_{model_tag(args.model)}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    for length_key, item in summaries.items():
        print(f"{args.model} L{length_key}: G_m={item['G_m']:.6f}, std={item['case_gap_std']:.6f}")


if __name__ == "__main__":
    run(parse_args())

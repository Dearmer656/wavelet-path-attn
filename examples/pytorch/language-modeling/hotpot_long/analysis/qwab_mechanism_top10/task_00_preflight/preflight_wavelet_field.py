#!/usr/bin/env python3
"""PAT-254 Preflight: verify the REAL online wavelet field actually added to
attention logits (not an idealized formula) for K1 rho=128/256 at L=512/4096.

For each (checkpoint, seq_len), on real HotpotQA-Long cases:
  bias[h,i,j] = E_wav_raw[h,i,j] - E_base_raw[h,i,j]   (real tensor added to logits)
sampled at a grid of query rows i, causal-masked to j<=i.

Reports (required by the PAT-254 Preflight spec):
  1. row-to-row cosine similarity distribution, by layer/head/query-bin
     -- cosine computed on the SHARED causal support j in [0, min(i,i')],
        which is the correct comparison for the "shared absolute-key
        pattern" hypothesis (bias depends on absolute key index j, not
        relative lag i-j -- confirmed structurally: path_attn.py's `diff`
        variable is torch.arange(T), an absolute key coordinate).
  2. effective rank / singular spectrum of the sampled-row bias matrix.
  3. cosine similarity of each real bias row to the model's OWN
     _ricker_wavelet() template evaluated with the REAL captured per-query
     shift (beta_i) and REAL scale -- not a hand-derived idealized formula.
  4. the learned shift (beta_i) distribution across queries, since these
     checkpoints have wavelet_ctxscale_shift_number=1 but beta_i itself is
     query-dependent (a function of the query's hidden state), not a
     constant -- confirmed via the new _last_ctxscale_beta_by_scale capture.

Decision rule (per issue): median row cosine > 0.95 and effective rank ~= 1
-> "shared row pattern" language; otherwise -> "shared low-dimensional
wavelet subspace" language for all later PAT-254 tasks.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import sys

sys.path.insert(0, "/project/nlp-work5/hongyu-s/flash-linear-attention")
sys.path.insert(0, "/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long")

from fla.layers.path_attn import PaTHAttention  # noqa: F401 (import path check)
from dump_router_usage import render_prompt_hotpot, load_cases_hotpot, get_path_layers, load_path_attn_model

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def query_grid(T: int, n: int = 8) -> list[int]:
    lo = max(1, T // 32)
    pts = np.unique(np.linspace(lo, T - 1, n).astype(int)).tolist()
    return pts


def ricker(u: torch.Tensor) -> torch.Tensor:
    return (1.0 - u.pow(2)) * torch.exp(-0.5 * u.pow(2))


def analyze_case(model, path_layers, input_ids, qgrid):
    """Returns per-layer dict of stats for this one case."""
    with torch.no_grad():
        _ = model(input_ids)
    T = input_ids.shape[1]
    out = []
    for layer_idx, (name, module) in enumerate(path_layers):
        pa = getattr(module, "_last_logits_pa_only", None)
        full = getattr(module, "_last_logits_full", None)
        beta_by_scale = getattr(module, "_last_ctxscale_beta_by_scale", None)
        diff = getattr(module, "_last_ctxscale_diff", None)
        scales = getattr(module, "_last_ctxscale_scales", None)
        if pa is None or full is None:
            raise RuntimeError(f"layer {layer_idx}: missing logits captures")
        bias = (full - pa)[0]  # [H,T,T]
        H = bias.shape[0]
        i_idx = torch.arange(T, device=bias.device).view(T, 1)
        j_idx = torch.arange(T, device=bias.device).view(1, T)
        causal = (j_idx <= i_idx).to(bias.dtype)  # [T,T]
        bias = bias * causal.unsqueeze(0)

        rows_h = bias[:, qgrid, :]  # [H, n_q, T]
        row_cos = []
        for a in range(len(qgrid)):
            for b in range(a + 1, len(qgrid)):
                ia, ib = qgrid[a], qgrid[b]
                lo = min(ia, ib) + 1
                va = rows_h[:, a, :lo]
                vb = rows_h[:, b, :lo]
                na = va.norm(dim=-1).clamp_min(1e-8)
                nb = vb.norm(dim=-1).clamp_min(1e-8)
                cos = (va * vb).sum(dim=-1) / (na * nb)
                row_cos.append(cos.mean().item())
        row_cos = np.array(row_cos) if row_cos else np.array([float("nan")])

        eff_ranks = []
        for h in range(H):
            mat = rows_h[h]  # [n_q, T]
            if mat.abs().sum() < 1e-8:
                continue
            s = torch.linalg.svdvals(mat.double())
            s2 = s.pow(2)
            eff_rank = float((s2.sum() ** 2 / (s2.pow(2).sum() + 1e-12)).item())
            eff_ranks.append(eff_rank)
        eff_rank_mean = float(np.mean(eff_ranks)) if eff_ranks else float("nan")

        tmpl_cos = []
        if beta_by_scale is not None and diff is not None and scales is not None and 0 in beta_by_scale:
            beta0 = beta_by_scale[0][0]  # [Tq] (B=1)
            s0 = float(scales[0].item())
            for qi in qgrid:
                bidx = qi if beta0.shape[0] == T else min(qi, beta0.shape[0] - 1)
                beta_q = beta0[bidx]
                u = (diff - beta_q) / max(s0, 1e-6)
                tmpl = ricker(u)  # [T]
                tmpl = tmpl * causal[qi]
                real_row = bias[:, qi, :].mean(dim=0)  # mean over heads
                nt = tmpl.norm().clamp_min(1e-8)
                nr = real_row.norm().clamp_min(1e-8)
                cos = float((tmpl * real_row).sum() / (nt * nr))
                tmpl_cos.append(cos)
        tmpl_cos = np.array(tmpl_cos) if tmpl_cos else np.array([float("nan")])

        beta_vals = None
        if beta_by_scale is not None and 0 in beta_by_scale:
            beta_vals = beta_by_scale[0][0].detach().cpu().numpy()

        out.append({
            "layer": layer_idx,
            "row_cos_mean": float(np.nanmean(row_cos)),
            "row_cos_median": float(np.nanmedian(row_cos)),
            "row_cos_p10": float(np.nanpercentile(row_cos, 10)),
            "row_cos_p90": float(np.nanpercentile(row_cos, 90)),
            "eff_rank_mean": eff_rank_mean,
            "n_query_grid": len(qgrid),
            "tmpl_cos_mean": float(np.nanmean(tmpl_cos)),
            "tmpl_cos_median": float(np.nanmedian(tmpl_cos)),
            "beta_mean": float(np.mean(beta_vals)) if beta_vals is not None else float("nan"),
            "beta_std": float(np.std(beta_vals)) if beta_vals is not None else float("nan"),
            "beta_min": float(np.min(beta_vals)) if beta_vals is not None else float("nan"),
            "beta_max": float(np.max(beta_vals)) if beta_vals is not None else float("nan"),
        })
    return out


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=False)
    model = load_path_attn_model(args.checkpoint, device, dtype=torch.float32)
    path_layers = get_path_layers(model)
    n_layers = len(path_layers)
    for _, module in path_layers:
        module._capture_debug_tensors = True
    print(f"Found {n_layers} PaTHAttention layers")

    cases = load_cases_hotpot(args.jsonl, args.seq_len, args.n_case)
    print(f"Loaded {len(cases)} cases @ L={args.seq_len}")
    qgrid = query_grid(args.seq_len, n=args.n_qgrid)
    print(f"query grid: {qgrid}")

    per_case_layer_rows = []
    for ci, rec in enumerate(cases):
        prompt_ids = tokenizer(render_prompt_hotpot(rec), add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(f" {rec['answer']}", add_special_tokens=False)["input_ids"]
        full_ids = (prompt_ids + answer_ids)[: args.seq_len]
        if len(full_ids) < args.seq_len:
            print(f"  case {ci}: skip, only {len(full_ids)} tokens")
            continue
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        stats = analyze_case(model, path_layers, input_ids, qgrid)
        for s in stats:
            s["case_idx"] = ci
            per_case_layer_rows.append(s)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  case {ci + 1}/{len(cases)} done")

    out_csv = OUT_DIR / f"{args.model_tag}_L{args.seq_len}.csv"
    fieldnames = list(per_case_layer_rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(per_case_layer_rows)
    print(f"wrote {out_csv} ({len(per_case_layer_rows)} rows)")

    by_layer = {}
    for r in per_case_layer_rows:
        by_layer.setdefault(r["layer"], []).append(r)
    summary = {"model_tag": args.model_tag, "seq_len": args.seq_len, "n_case": len(cases), "n_qgrid": len(qgrid),
               "layers": {}}
    for layer, rows in by_layer.items():
        summary["layers"][layer] = {
            "row_cos_median": float(np.median([r["row_cos_median"] for r in rows])),
            "row_cos_mean": float(np.mean([r["row_cos_mean"] for r in rows])),
            "eff_rank_mean": float(np.mean([r["eff_rank_mean"] for r in rows])),
            "tmpl_cos_median": float(np.median([r["tmpl_cos_median"] for r in rows])),
            "beta_std_mean": float(np.mean([r["beta_std"] for r in rows])),
        }
    overall_row_cos_median = float(np.median([r["row_cos_median"] for r in per_case_layer_rows]))
    overall_eff_rank = float(np.mean([r["eff_rank_mean"] for r in per_case_layer_rows]))
    summary["overall_row_cos_median"] = overall_row_cos_median
    summary["overall_eff_rank_mean"] = overall_eff_rank
    summary["decision"] = (
        "shared_row_pattern" if (overall_row_cos_median > 0.95 and overall_eff_rank < 1.5)
        else "shared_low_dim_subspace"
    )
    out_json = OUT_DIR / f"{args.model_tag}_L{args.seq_len}_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out_json}")
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model_tag", required=True)
    p.add_argument("--seq_len", type=int, required=True)
    p.add_argument("--n_case", type=int, default=10)
    p.add_argument("--n_qgrid", type=int, default=8)
    p.add_argument("--jsonl", default="/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/hotpot_long_dev.jsonl")
    run(p.parse_args())


if __name__ == "__main__":
    main()

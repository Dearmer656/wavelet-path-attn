#!/usr/bin/env python3
"""hotpot_free_gen_eval.py — PAT-217: open-ended (free-generation) HotpotQA eval.

Unlike the existing teacher-forced eval (Trainer.evaluate() with gold answer in input_ids),
this script renders only the PROMPT (context + question + "Answer:"), runs model.generate(),
then computes F1/EM against gold answer + answer_aliases.

Motif substitution is applied via the same MOTIF_* env vars as motif_launcher.py.
Supports multi-GPU via torchrun (each rank handles a slice of examples).

Usage (inference-only, no motif):
  python hotpot_free_gen_eval.py --checkpoint <ckpt> --jsonl <jsonl> --L 2048

Usage (with motif, via torchrun):
  MOTIF_MODE=real MOTIF_NPZ=... MOTIF_DIM_SELECTIVE=1 MOTIF_LAYERS=0-1-...-11 \\
  python -m torch.distributed.run --nproc_per_node=4 \\
    hotpot_free_gen_eval.py --checkpoint <ckpt> --jsonl <jsonl> --L 2048
"""
import argparse, json, os, re, string, unicodedata
import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


# ── F1 / EM helpers (same as run_clm.py) ────────────────────────────────────
def _normalize(s):
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c if c not in string.punctuation else " " for c in s)
    return " ".join(s.split())


def _f1(pred, gold):
    p_toks = _normalize(pred).split()
    g_toks = _normalize(gold).split()
    if not p_toks or not g_toks:
        return int(p_toks == g_toks)
    common = sum((min(p_toks.count(t), g_toks.count(t)) for t in set(g_toks)))
    if common == 0:
        return 0.0
    prec = common / len(p_toks)
    rec  = common / len(g_toks)
    return 2 * prec * rec / (prec + rec)


# ── prompt rendering (matches distill_bias_step1.py render_trained) ──────────
def render_prompt(ex):
    lines = [f"{t}: {s}" for t, sents in ex["context"] for s in sents]
    return f"Context:\n{chr(10).join(lines)}\nQuestion: {ex['question']}\nAnswer:"


# ── motif substitution (env-gated, mirrors motif_launcher.py) ────────────────
def _apply_motif_if_configured(model, nl, nh):
    mode = os.environ.get("MOTIF_MODE", "off")
    if mode == "off":
        print("[free-gen] MOTIF_MODE=off — vanilla model", flush=True)
        return
    npz  = os.environ.get("MOTIF_NPZ", "")
    lam  = float(os.environ.get("MOTIF_LAM", "1.0"))
    tail = os.environ.get("MOTIF_TAIL", "hold_last")
    dim_sel  = os.environ.get("MOTIF_DIM_SELECTIVE", "0") == "1"
    pcut     = float(os.environ.get("MOTIF_PERIOD_CUTOFF", "512"))
    min_ve   = float(os.environ.get("MOTIF_MIN_TRAIN_VE", "0.3"))
    lam_mode = os.environ.get("MOTIF_LAM_MODE", "const")

    import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from motif_substitution import (apply_motif_substitution, select_broken_heads,
                                    compute_adaptive_lam)

    layers_env = os.environ.get("MOTIF_LAYERS", "")
    csv_path   = os.environ.get("MOTIF_RECON_CSV", "")
    topk       = int(os.environ.get("MOTIF_TOPK", "16"))

    if layers_env:
        layers = [int(x) for x in re.split(r"[^0-9]+", layers_env) if x]
        broken = [(l, h) for l in layers for h in range(nh)]
        print(f"[free-gen] MOTIF_LAYERS={layers} → {len(broken)} heads", flush=True)
    else:
        broken = select_broken_heads(csv_path, topk, npz_path=npz, min_train_ve=min_ve, nh=nh)
        print(f"[free-gen] top-{topk} broken heads: {broken}", flush=True)

    lam_val = (compute_adaptive_lam(csv_path, npz, broken, lam_base=lam, mode="recon",
                                    min_train_ve=min_ve, nl=nl, nh=nh)
               if lam_mode == "recon" else lam)

    apply_motif_substitution(model, npz, broken, lam=lam_val, mode=mode,
                             nl=nl, nh=nh, tail=tail,
                             dim_selective=dim_sel, period_cutoff=pcut)


# ── main ─────────────────────────────────────────────────────────────────────
def run(a):
    # distributed setup
    rank, world = 0, 1
    if dist.is_available() and int(os.environ.get("WORLD_SIZE", 1)) > 1:
        dist.init_process_group("nccl")
        rank  = dist.get_rank()
        world = dist.get_world_size()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    if rank == 0:
        print(f"[free-gen] L={a.L} ckpt={a.checkpoint} world={world} "
              f"max_new_tokens={a.max_new_tokens} n_cases={a.n_cases}", flush=True)

    # load model
    try:
        tok = AutoTokenizer.from_pretrained(a.checkpoint, use_fast=True); tok(["x"])
    except Exception:
        tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    cfg = AutoConfig.from_pretrained(a.checkpoint)
    cfg.attn_implementation = "eager"; cfg.pe_method = "rotary"
    model = AutoModelForCausalLM.from_pretrained(
        a.checkpoint, config=cfg, torch_dtype=torch.float32,
        trust_remote_code=True).eval().to(device)

    nl = int(getattr(cfg, "n_layer", 12))
    nh = int(getattr(cfg, "n_head", 12))
    _apply_motif_if_configured(model, nl, nh)

    # load examples
    cases = []
    with open(a.jsonl) as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            if int(r.get("meta", {}).get("target_total_tokens", -1)) != a.L:
                continue
            cases.append(r)
            if a.n_cases > 0 and len(cases) >= a.n_cases:
                break
    if rank == 0:
        print(f"[free-gen] loaded {len(cases)} cases at L={a.L}", flush=True)

    # prompt token budget: leave room for generated tokens
    prompt_cap = a.L - a.max_new_tokens

    gen_kwargs = dict(
        max_new_tokens=a.max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        use_cache=True,                  # KV-cache for speed; motif decode branch handles this
    )

    # split across ranks
    my_cases = cases[rank::world]
    records  = []

    for ci, ex in enumerate(my_cases):
        prompt = render_prompt(ex)
        ids    = tok(prompt, add_special_tokens=True, truncation=False)["input_ids"]
        # left-truncate if prompt > budget (keep the tail = most recent context)
        if len(ids) > prompt_cap:
            ids = ids[-prompt_cap:]

        input_t = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(input_t, **gen_kwargs)
        gen_ids  = out[0][input_t.shape[1]:].tolist()
        pred     = tok.decode(gen_ids, skip_special_tokens=True).strip()

        golds = [ex.get("answer", "")]
        golds += [str(a) for a in ex.get("answer_aliases", []) if str(a).strip()]
        golds = [g for g in golds if g.strip()] or [""]

        f1_val = max(_f1(pred, g)  for g in golds)
        em_val = max(float(_normalize(pred) == _normalize(g)) for g in golds)
        records.append({"f1": f1_val, "em": em_val, "pred": pred,
                        "gold": golds[0], "case_id": ex.get("_id", ci)})

        if rank == 0 and (ci + 1) % 50 == 0:
            f1s = [r["f1"] for r in records]
            print(f"[free-gen] {ci+1}/{len(my_cases)}  running F1={np.mean(f1s):.4f}", flush=True)

    # gather across ranks
    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, records)
        if rank == 0:
            records = [r for shard in gathered for r in shard]
    else:
        pass  # records already complete

    if rank == 0:
        f1s = [r["f1"] for r in records]
        ems = [r["em"] for r in records]
        print(f"\n=== FREE-GEN EVAL L={a.L} n={len(records)} ===")
        print(f"  F1 = {np.mean(f1s):.4f}   EM = {np.mean(ems):.4f}")
        print(f"  (teacher-forced ref: baseline 0.068, motif-infer 0.657)")

        if a.out:
            os.makedirs(a.out, exist_ok=True)
            out_path = os.path.join(a.out, f"free_gen_L{a.L}_results.json")
            with open(out_path, "w") as f:
                json.dump({"f1": float(np.mean(f1s)), "em": float(np.mean(ems)),
                           "n": len(records), "L": a.L,
                           "records": records[:200]}, f, indent=2)
            print(f"  saved -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--jsonl",       required=True)
    ap.add_argument("--L",           type=int, default=2048)
    ap.add_argument("--n_cases",     type=int, default=500,
                    help="max examples per rank (0=all)")
    ap.add_argument("--max_new_tokens", type=int, default=50)
    ap.add_argument("--out",         default="")
    run(ap.parse_args())

#!/usr/bin/env python3
"""motif_gen_debug.py — Link-B error hunt: is the motif substitution ACTUALLY active
during answer generation (the thing that determines HotpotQA F1)?

Loads the finetuned real ckpt, applies substitution (mode=real), and calls model.generate()
with use_cache=False (matches --eval_generate_no_cache True) and use_cache=True, with
MOTIF_DEBUG=1, so the patched eager prints q_len/k_len and whether the bias was injected at
every decode step. If injection is NOT happening during generation, the L3 null was a no-op
artifact, not a refutation.
"""
import os
os.environ["MOTIF_DEBUG"] = "1"
import json
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import motif_substitution as MS
from motif_substitution import apply_motif_substitution, select_broken_heads


def render_trained(ex):
    lines = [f"{t}: {s}" for t, sents in ex["context"] for s in sents]
    return f"Context:\n{chr(10).join(lines)}\nQuestion: {ex['question']}\nAnswer:"


def run(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(a.checkpoint)
    cfg.attn_implementation = "eager"; cfg.pe_method = "rotary"
    model = AutoModelForCausalLM.from_pretrained(
        a.checkpoint, config=cfg, torch_dtype=torch.float32, trust_remote_code=True).eval().to(dev)
    broken = select_broken_heads(a.recon_csv, 16, npz_path=a.npz, min_train_ve=0.3)
    apply_motif_substitution(model, a.npz, broken, lam=1.0, mode="real")

    rec = None
    for line in open(a.jsonl):
        if line.strip():
            r = json.loads(line)
            if int(r.get("meta", {}).get("target_total_tokens", -1)) == a.L:
                rec = r; break
    ids = tok(render_trained(rec), add_special_tokens=False)["input_ids"][:a.L - 12]
    x = torch.tensor([ids], device=dev)
    print(f"prompt_len={len(ids)}", flush=True)

    for uc in (False, True):
        MS._dbg.update(n=0, inj=0, seen=0)
        print(f"\n===== generate use_cache={uc} =====", flush=True)
        with torch.no_grad():
            model.generate(x, max_new_tokens=6, do_sample=False,
                           use_cache=uc, pad_token_id=tok.pad_token_id)
        print(f"[SUMMARY use_cache={uc}] eager calls seen={MS._dbg['seen']} "
              f"bias-injected={MS._dbg['inj']} "
              f"({'ALL injected' if MS._dbg['inj']==MS._dbg['seen'] else 'SOME/NONE SKIPPED'})",
              flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--recon_csv", required=True)
    ap.add_argument("--L", type=int, default=2048)
    run(ap.parse_args())

#!/usr/bin/env python3
"""ruler_motif_eval.py — PAT-222: RULER eval on Llama-style LLMs with motif injection.

Runs open-ended generation on RULER tasks (retrieval + variable tracking) at controlled
lengths. Motif is injected via patched LlamaAttention.forward (use_cache=False for
correctness, as specified in PAT-222).

The LlamaAttention.forward is replaced per-layer to add s(δ)+v(j) positional bias
after the QK matmul and before softmax. GQA is handled: bias shape is [1, nh, q, k]
matching the post-repeat_kv attention logits.

Usage:
  # baseline (no motif):
  python ruler_motif_eval.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
      --jsonl data/ruler_eval_4096.jsonl --L 4096 --out analysis_outputs/pat222/ruler

  # with motif:
  MOTIF_NPZ=analysis_outputs/pat222/tinyllama/llm_motif_L2048.npz \\
  python ruler_motif_eval.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
      --jsonl data/ruler_eval_4096.jsonl --L 4096 --out analysis_outputs/pat222/ruler
"""
import argparse, json, math, os, re, string
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import transformers.models.llama.modeling_llama as _M
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import repeat_kv


# ── bias helpers (reuse PAT-217 logic) ───────────────────────────────────────
def _extend_s(s_row, L0, T, tail="hold_last"):
    s = np.empty(T, np.float32); s[:min(L0, T)] = s_row[:min(L0, T)]
    if T > L0:
        if tail == "hold_last":
            s[L0:] = s_row[L0 - 1]
        else:
            s[L0:] = 0.0
    return s


def _extend_v(v_row, L0, T):
    v = np.empty(T, np.float32); v[:min(L0, T)] = v_row[:min(L0, T)]
    if T > L0:
        v[L0:] = float(np.median(v_row[64:L0]) if L0 > 64 else 0.0)
    return v


def _build_bias_matrix(s_tab, v_tab, nl, nh, layer_idx, T, device, dtype, lam=1.0):
    """[1, nh, T, T] bias for full prefill."""
    base = layer_idx * nh
    off = np.clip(np.arange(T)[:, None] - np.arange(T)[None, :], 0, T - 1)
    B = np.zeros((nh, T, T), np.float32)
    L0 = s_tab.shape[1]
    for h in range(nh):
        s = _extend_s(s_tab[base + h], L0, T)
        v = _extend_v(v_tab[base + h], L0, T)
        B[h] = lam * (s[off] + v[None, :])
    return torch.from_numpy(B).unsqueeze(0).to(device=device, dtype=dtype)


def _build_bias_row(s_tab, v_tab, nl, nh, layer_idx, pos, k_len, device, dtype, lam=1.0):
    """[1, nh, 1, k_len] bias for a single query at absolute position pos (decode)."""
    base = layer_idx * nh
    j = np.arange(k_len)
    deltas = np.clip(pos - j, 0, k_len - 1)
    B = np.zeros((nh, 1, k_len), np.float32)
    L0 = s_tab.shape[1]
    for h in range(nh):
        s = _extend_s(s_tab[base + h], L0, k_len)
        v = _extend_v(v_tab[base + h], L0, k_len)
        B[h, 0] = lam * (s[deltas] + v[j])
    return torch.from_numpy(B).unsqueeze(0).to(device=device, dtype=dtype)


# ── Llama attention patching ──────────────────────────────────────────────────
_MOTIF_STATE = {}   # populated by patch_llama_attention; read inside patched eager_attn


def patch_llama_attention(model, s_tab, v_tab, nl, nh, lam=1.0, use_cache_inject=False):
    """Patch eager_attention_forward in the Llama modeling module to inject motif bias.

    Works with both old-style (rotary on attn) and new-style (rotary at model level)
    Llama implementations — injection is at the standalone eager_attention_forward fn.
    """
    _MOTIF_STATE["s_tab"] = s_tab
    _MOTIF_STATE["v_tab"] = v_tab
    _MOTIF_STATE["nl"]    = nl
    _MOTIF_STATE["nh"]    = nh
    _MOTIF_STATE["lam"]   = lam
    _MOTIF_STATE["use_cache_inject"] = use_cache_inject

    import transformers.models.llama.modeling_llama as _LM
    _orig = _LM.eager_attention_forward

    def patched_eager(module, query, key, value, attention_mask,
                      scaling, dropout=0.0, **kwargs):
        st    = _MOTIF_STATE
        nh_   = st["nh"]; nl_ = st["nl"]
        s_    = st["s_tab"]; v_ = st["v_tab"]; lam_ = st["lam"]

        key_states   = repeat_kv(key,   module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

        # ── motif injection ──────────────────────────────────────────────────
        li    = module.layer_idx
        q_len = query.shape[-2]
        k_len = key_states.shape[-2]
        dev, dt = query.device, query.dtype

        if st["use_cache_inject"] and q_len == 1 and k_len > 1:
            pos  = k_len - 1
            bias = _build_bias_row(s_, v_, nl_, nh_, li, pos, k_len, dev, dt, lam_)
        else:
            bias = _build_bias_matrix(s_, v_, nl_, nh_, li, k_len, dev, dt, lam_)
            if q_len < k_len:
                bias = bias[:, :, k_len - q_len:, :]
        attn_weights = attn_weights + bias
        # ────────────────────────────────────────────────────────────────────

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
        attn_output  = torch.matmul(attn_weights, value_states)
        attn_output  = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    _LM.eager_attention_forward = patched_eager


# ── RULER scoring ─────────────────────────────────────────────────────────────
def _normalize(s):
    return " ".join(re.sub(r"[%s]" % re.escape(string.punctuation), " ", s.lower()).split())


def ruler_score(pred, gold_list):
    """1.0 if any gold answer appears as a word in pred (RULER-style)."""
    pred_n = _normalize(pred)
    for g in gold_list:
        g_n = _normalize(g)
        if g_n and g_n in pred_n:
            return 1.0
    return 0.0


# ── main ─────────────────────────────────────────────────────────────────────
def run(a):
    rank, world = 0, 1
    if dist.is_available() and int(os.environ.get("WORLD_SIZE", 1)) > 1:
        dist.init_process_group("nccl")
        rank = dist.get_rank(); world = dist.get_world_size()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    if rank == 0:
        print(f"[ruler-eval] model={a.model} L={a.L} world={world} motif_npz={a.motif_npz}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model, cache_dir=a.cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    cfg = AutoConfig.from_pretrained(a.model, cache_dir=a.cache_dir)
    cfg._attn_implementation = "eager"   # always eager so patch takes effect
    model = AutoModelForCausalLM.from_pretrained(
        a.model, config=cfg, torch_dtype=torch.float32,
        cache_dir=a.cache_dir, trust_remote_code=True).eval().to(device)

    nl  = cfg.num_hidden_layers
    nh  = cfg.num_attention_heads

    # load + apply motif
    if a.motif_npz:
        d = np.load(a.motif_npz)
        s_tab, v_tab = d["s"], d["v"]
        patch_llama_attention(model, s_tab, v_tab, nl, nh, lam=a.lam,
                              use_cache_inject=not a.no_cache)
        if rank == 0:
            print(f"[ruler-eval] motif injected from {a.motif_npz}", flush=True)

    # load cases
    cases = []
    with open(a.jsonl) as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            if int(r.get("length", -1)) != a.L:
                continue
            cases.append(r)
            if a.n_cases > 0 and len(cases) >= a.n_cases:
                break
    if rank == 0:
        print(f"[ruler-eval] {len(cases)} cases at L={a.L}", flush=True)

    # prompt budget (leave room for generated answer)
    prompt_cap = a.L - a.max_new_tokens

    gen_kwargs = dict(
        max_new_tokens=a.max_new_tokens,
        do_sample=False, num_beams=1,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        use_cache=not a.no_cache,
    )

    my_cases = cases[rank::world]
    records  = []

    for ci, ex in enumerate(my_cases):
        raw_input = ex["input"]
        gold_list = ex["outputs"]       # list of acceptable answers
        task      = ex.get("ruler_config", "unknown")

        # apply chat template if available
        if hasattr(tok, "apply_chat_template") and tok.chat_template:
            msgs = [{"role": "user", "content": raw_input}]
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        else:
            prompt = raw_input

        ids = tok(prompt, add_special_tokens=False, truncation=False)["input_ids"]
        if len(ids) > prompt_cap:
            ids = ids[-prompt_cap:]     # keep tail (question is at the end)

        input_t = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(input_t, **gen_kwargs)
        gen_ids = out[0][input_t.shape[1]:].tolist()
        pred    = tok.decode(gen_ids, skip_special_tokens=True).strip()

        score = ruler_score(pred, gold_list)
        records.append({"score": score, "task": task, "pred": pred,
                        "gold": gold_list[0], "case_idx": ci})

        if rank == 0 and (ci + 1) % 20 == 0:
            acc = np.mean([r["score"] for r in records])
            print(f"[ruler-eval] {ci+1}/{len(my_cases)}  acc={acc:.4f}", flush=True)

    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, records)
        if rank == 0:
            records = [r for shard in gathered for r in shard]

    if rank == 0:
        # overall
        acc_all = np.mean([r["score"] for r in records])
        print(f"\n=== RULER EVAL L={a.L}  n={len(records)} ===")
        print(f"  Overall accuracy: {acc_all:.4f}")
        # per-task
        from collections import defaultdict
        by_task = defaultdict(list)
        for r in records:
            by_task[r["task"]].append(r["score"])
        for t, scores in sorted(by_task.items()):
            print(f"  {t:40s}  acc={np.mean(scores):.4f}  n={len(scores)}")

        os.makedirs(a.out, exist_ok=True)
        tag = "motif" if a.motif_npz else "baseline"
        out_json = os.path.join(a.out, f"ruler_{tag}_L{a.L}.json")
        with open(out_json, "w") as f:
            json.dump({"acc": float(acc_all), "L": a.L, "n": len(records),
                       "by_task": {t: float(np.mean(s)) for t, s in by_task.items()},
                       "records": records[:200]}, f, indent=2)
        print(f"  saved → {out_json}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",         default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--jsonl",         required=True)
    ap.add_argument("--L",             type=int, default=4096)
    ap.add_argument("--n_cases",       type=int, default=0, help="0=all")
    ap.add_argument("--max_new_tokens",type=int, default=50)
    ap.add_argument("--motif_npz",     default="", help="path to llm_motif_L*.npz; empty=baseline")
    ap.add_argument("--lam",           type=float, default=1.0)
    ap.add_argument("--no_cache",      action="store_true", help="use_cache=False for generation")
    ap.add_argument("--out",           default="analysis_outputs/pat222/ruler")
    ap.add_argument("--cache_dir",     default="/cl/work5/hongyu-s/huggingfac")
    run(ap.parse_args())

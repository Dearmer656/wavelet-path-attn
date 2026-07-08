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


# ── bias helpers — GPU-native, no large numpy arrays ─────────────────────────

def _extend_s_np(s_row, L0, T):
    """1-D numpy extend (cheap, O(T) not O(T^2))."""
    if T <= L0:
        return s_row[:T]
    out = np.empty(T, np.float32)
    out[:L0] = s_row; out[L0:] = s_row[L0 - 1]
    return out


def _extend_v_np(v_row, L0, T):
    if T <= L0:
        return v_row[:T]
    out = np.empty(T, np.float32)
    out[:L0] = v_row
    out[L0:] = float(np.median(v_row[64:L0]) if L0 > 64 else 0.0)
    return out


def _build_bias_matrix(s_tab, v_tab, nl, nh, layer_idx, T, device, dtype, lam=1.0):
    """[1, nh, T, T] bias — all heavy work on GPU, only O(nh*T) numpy transfer."""
    base = layer_idx * nh
    L0   = s_tab.shape[1]
    # Upload 1-D s/v vectors per head (cheap: nh*T*4 bytes << T^2*4 bytes)
    s_np = np.stack([_extend_s_np(s_tab[base + h], L0, T) for h in range(nh)])  # [nh, T]
    v_np = np.stack([_extend_v_np(v_tab[base + h], L0, T) for h in range(nh)])  # [nh, T]
    s_t  = torch.from_numpy(s_np).to(device=device, dtype=dtype)  # [nh, T]
    v_t  = torch.from_numpy(v_np).to(device=device, dtype=dtype)  # [nh, T]
    # Build O(T^2) on GPU via broadcast gather (single kernel, fast)
    idx  = torch.arange(T, device=device)
    off  = (idx.unsqueeze(1) - idx.unsqueeze(0)).clamp(0, T - 1)  # [T, T]
    bias = lam * (s_t[:, off] + v_t.unsqueeze(1))                  # [nh, T, T]
    return bias.unsqueeze(0)                                        # [1, nh, T, T]


def _build_bias_row(s_tab, v_tab, nl, nh, layer_idx, pos, k_len, device, dtype, lam=1.0):
    """[1, nh, 1, k_len] bias for a single query at absolute position pos."""
    base = layer_idx * nh
    L0   = s_tab.shape[1]
    s_np = np.stack([_extend_s_np(s_tab[base + h], L0, k_len) for h in range(nh)])  # [nh, k_len]
    v_np = np.stack([_extend_v_np(v_tab[base + h], L0, k_len) for h in range(nh)])  # [nh, k_len]
    s_t  = torch.from_numpy(s_np).to(device=device, dtype=dtype)
    v_t  = torch.from_numpy(v_np).to(device=device, dtype=dtype)
    j    = torch.arange(k_len, device=device)
    d    = (pos - j).clamp(0, k_len - 1)                          # [k_len]
    bias = lam * (s_t[:, d] + v_t)                                 # [nh, k_len]
    return bias.unsqueeze(0).unsqueeze(2)                           # [1, nh, 1, k_len]


# ── Llama attention patching ──────────────────────────────────────────────────
_MOTIF_STATE = {}   # populated by patch_llama_attention; read inside patched eager_attn

# Side-channel for pre-RoPE Q/K capture (used to remove broken low-freq rotation)
_PRE_ROPE_CACHE = {}   # layer_idx -> (q_pre, k_pre)
_PRE_LAYER_CTR  = [0]


def _install_pre_rope_hook(model):
    """Capture pre-rotation Q/K per layer so we can zero out low-freq RoPE in patched_eager."""
    import transformers.models.llama.modeling_llama as _LM

    # Reset counter at each model.forward() call (each decode step with use_cache=False)
    def _reset_hook(module, inp):
        _PRE_LAYER_CTR[0] = 0
        _PRE_ROPE_CACHE.clear()
    model.model.register_forward_pre_hook(_reset_hook)

    orig_rotary = _LM.apply_rotary_pos_emb
    def _capture_rotary(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        li = _PRE_LAYER_CTR[0]
        _PRE_ROPE_CACHE[li] = (q, k)   # GPU tensors, not copied
        q_r, k_r = orig_rotary(q, k, cos, sin, position_ids, unsqueeze_dim)
        _PRE_LAYER_CTR[0] += 1
        return q_r, k_r
    _LM.apply_rotary_pos_emb = _capture_rotary


def patch_llama_attention(model, s_tab, v_tab, nl, nh, lf_mask_np,
                          lam=1.0, use_cache_inject=False):
    """Patch eager_attention_forward to inject motif with NoPE-lf substitution.

    For low-freq RoPE dims (period > period_cutoff), we replace post-RoPE Q/K with their
    pre-rotation counterparts — removing the broken OOD low-freq rotation.  The motif bias
    (distilled from B_rope_lf - B_nope_lf at training length) is then added on top, giving:
        attn_logits = B_rope_hf  +  B_nope_lf  +  motif
    instead of the flawed:
        attn_logits = B_rope_hf  +  B_rope_lf_ood(broken)  +  motif
    """
    _MOTIF_STATE["s_tab"]            = s_tab
    _MOTIF_STATE["v_tab"]            = v_tab
    _MOTIF_STATE["nl"]               = nl
    _MOTIF_STATE["nh"]               = nh
    _MOTIF_STATE["lam"]              = lam
    _MOTIF_STATE["use_cache_inject"] = use_cache_inject
    lf_t = torch.from_numpy(lf_mask_np)
    _MOTIF_STATE["lf_f"]  = lf_t.float()          # [head_dim] float  1=low-freq
    _MOTIF_STATE["hf_f"]  = (1.0 - lf_t).float()  # [head_dim] float  1=high-freq

    _install_pre_rope_hook(model)

    import transformers.models.llama.modeling_llama as _LM
    _orig = _LM.eager_attention_forward

    def patched_eager(module, query, key, value, attention_mask,
                      scaling, dropout=0.0, **kwargs):
        st    = _MOTIF_STATE
        nh_   = st["nh"]; nl_ = st["nl"]
        s_    = st["s_tab"]; v_ = st["v_tab"]; lam_ = st["lam"]
        dev   = query.device
        lf_f  = st["lf_f"].to(dev)   # [head_dim] float, 1=low-freq NoPE
        hf_f  = st["hf_f"].to(dev)   # [head_dim] float, 1=high-freq RoPE

        li    = module.layer_idx
        value_states = repeat_kv(value, module.num_key_value_groups)

        # ── NoPE substitution on low-freq dims ───────────────────────────────
        # For OOD lengths, low-freq RoPE phases are broken. Replace with pre-rotation
        # values via mask multiply (avoids slow CUDA boolean scatter/gather).
        # q_use = query * hf_mask + q_pre * lf_mask  (element-wise, coalesced)
        if li in _PRE_ROPE_CACHE:
            q_pre, k_pre = _PRE_ROPE_CACHE[li]
            q_use     = query * hf_f + q_pre * lf_f                                   # [B,nh,T,hd]
            k_pre_exp = k_pre.repeat_interleave(module.num_key_value_groups, dim=1)    # [B,nh,T,hd]
            k_use     = repeat_kv(key, module.num_key_value_groups) * hf_f + k_pre_exp * lf_f
        else:
            q_use = query
            k_use = repeat_kv(key, module.num_key_value_groups)
        # ────────────────────────────────────────────────────────────────────

        q_len = q_use.shape[-2]
        k_len = k_use.shape[-2]
        dev, dt = query.device, query.dtype

        attn_weights = torch.matmul(q_use, k_use.transpose(2, 3)) * scaling

        # ── motif injection ──────────────────────────────────────────────────
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
            causal_mask = attention_mask[:, :, :, :k_use.shape[-2]]
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

    # load + apply motif (with NoPE-lf substitution for broken OOD low-freq RoPE)
    if a.motif_npz:
        d = np.load(a.motif_npz)
        s_tab, v_tab = d["s"], d["v"]
        lf_mask_np   = d["dmask"].astype(np.float32)   # [head_dim], 1=low-freq
        patch_llama_attention(model, s_tab, v_tab, nl, nh, lf_mask_np,
                              lam=a.lam, use_cache_inject=not a.no_cache)
        if rank == 0:
            n_lf = int(lf_mask_np.sum())
            print(f"[ruler-eval] motif injected from {a.motif_npz} "
                  f"(NoPE-lf on {n_lf}/{len(lf_mask_np)} dims)", flush=True)

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

        # No chat template: RULER input is self-contained; chat template adds overhead
        # that forces truncation and removes needle records placed at the start.
        # No left-truncation: pass the full prompt so the model sees all records.
        # Positions beyond training length will be OOD — that is what we are testing.
        ids = tok(raw_input, add_special_tokens=True, truncation=False)["input_ids"]

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

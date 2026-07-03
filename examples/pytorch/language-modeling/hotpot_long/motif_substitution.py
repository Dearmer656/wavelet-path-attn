#!/usr/bin/env python3
"""motif_substitution.py — PAT-217 substitution mechanism (self-contained, no modeling edits).

On selected ("broken") heads, replace the RoPE logit with clean NoPE content-QK + a frozen
distilled structural motif b_h(i,j)=s_h(i-j)+v_h(j):
  (a) NoPE: patch rotate_queries_or_keys per selected head so its q/k stay UN-rotated.
  (b) motif bias: monkeypatch the module-level eager_attention_forward to inject a per-head
      frozen bias via the existing `qwab_bias` additive path (added to pre-softmax logits).
Non-selected heads keep qRk untouched. Control modes: real / shuffled / wrong_head / random.
"""
import numpy as np
import torch

_PATCHED = False


def make_masked_rotate(orig_rotate, nope_mask):
    H = int(nope_mask.numel())

    def wrapped(t, *a, **kw):
        rot = orig_rotate(t, *a, **kw)
        if t.dim() == 4 and t.shape[1] == H:
            m = nope_mask.to(device=rot.device, dtype=rot.dtype).view(1, H, 1, 1)
            return rot * (1.0 - m) + t * m
        return rot
    return wrapped


def _extend(s_row, v_row, L0, T):
    s = np.empty(T); s[:L0] = s_row; s[L0:] = s_row[L0 - 1]                  # slash tail hold
    base = np.median(v_row[64:L0]) if L0 > 64 else 0.0
    v = np.empty(T); v[:L0] = v_row; v[L0:] = base                          # sink baseline
    return s, v


def _build_layer_bias(s_lay, v_lay, hmask, lam, L0, T, device):
    """Return [1, nh, T, T] additive logit bias; zero on non-selected heads."""
    nh = s_lay.shape[0]
    off = (np.arange(T)[:, None] - np.arange(T)[None, :])                    # i-j, [T,T]
    off = np.clip(off, 0, T - 1)
    B = np.zeros((nh, T, T), dtype=np.float32)
    for h in range(nh):
        if hmask[h] <= 0:
            continue
        s, v = _extend(s_lay[h], v_lay[h], L0, T)
        B[h] = lam * (s[off] + v[None, :])                                  # s(i-j)+v(j)
    return torch.from_numpy(B).unsqueeze(0).to(device)                      # [1,nh,T,T]


def _install_eager_patch():
    global _PATCHED
    if _PATCHED:
        return
    import transformers.models.gpt2.modeling_gpt2 as M
    _orig = M.eager_attention_forward

    def patched(module, query, key, value, attention_mask, head_mask=None, **kwargs):
        sv = getattr(module, "_motif_sv", None)
        if sv is not None and kwargs.get("qwab_bias", None) is None:
            q_len, k_len = query.size(-2), key.size(-2)
            if q_len == k_len:                                             # skip KV-cache decode
                cache = getattr(module, "_motif_bias_cache", {})
                if q_len not in cache:
                    s_lay, v_lay, hmask, lam, L0 = sv
                    cache[q_len] = _build_layer_bias(s_lay, v_lay, hmask, lam, L0,
                                                     q_len, query.device)
                    module._motif_bias_cache = cache
                kwargs["qwab_bias"] = cache[q_len]
        return _orig(module, query, key, value, attention_mask, head_mask, **kwargs)

    M.eager_attention_forward = patched
    _PATCHED = True


def select_broken_heads(recon_csv, top_k):
    import csv
    rows = []
    with open(recon_csv) as f:
        for r in csv.DictReader(f):
            rows.append((int(r["layer"]), int(r["head"]), float(r["ood_rmse"])))
    rows.sort(key=lambda x: -x[2])
    return [(l, h) for l, h, _ in rows[:top_k]]


def apply_motif_substitution(model, npz_path, broken_heads, lam=1.0, mode="real",
                             nl=12, nh=12, seed=0):
    """broken_heads: list of (layer,head). mode in {real,shuffled,wrong_head,random,none}."""
    d = np.load(npz_path)
    S, V = d["s"].astype(np.float32), d["v"].astype(np.float32)   # [144, L0]
    L0 = int(d["L"])
    rng = np.random.default_rng(seed)

    # control transforms on the motif tables
    if mode == "shuffled":                       # shuffle δ-order within each row (destroys shape)
        for i in range(S.shape[0]):
            S[i] = S[i, rng.permutation(L0)]
    elif mode == "wrong_head":                   # assign each head a different head's motif
        perm = rng.permutation(S.shape[0]); S, V = S[perm], V[perm]
    elif mode == "random":                       # random nonneg motif of similar scale
        sc = np.abs(S).mean()
        S = rng.random(S.shape).astype(np.float32) * sc
        V = rng.random(V.shape).astype(np.float32) * np.abs(V).mean()

    per_layer_mask = {l: np.zeros(nh, np.float32) for l in range(nl)}
    for (l, h) in broken_heads:
        per_layer_mask[l][h] = 1.0

    _install_eager_patch()
    import re as _re
    n_heads = 0
    for name, mod in model.named_modules():
        m = _re.search(r"\.h\.(\d+)\.attn$", name)
        if m is None or not hasattr(mod, "rotary_emb"):
            continue
        l = int(m.group(1))
        hm = per_layer_mask[l]
        if hm.sum() == 0:
            continue
        # (a) NoPE on selected heads
        if not getattr(mod.rotary_emb, "_motif_patched", False):
            mod.rotary_emb.rotate_queries_or_keys = make_masked_rotate(
                mod.rotary_emb.rotate_queries_or_keys, torch.tensor(hm))
            mod.rotary_emb._motif_patched = True
        # (b) stash motif tables for this layer (used lazily in patched eager)
        if mode != "none":
            idx = [l * nh + h for h in range(nh)]
            mod._motif_sv = (S[idx], V[idx], hm, float(lam), L0)
            mod._motif_bias_cache = {}
        n_heads += int(hm.sum())
    print(f"[PAT-217] motif substitution mode={mode} lam={lam} on {n_heads} heads "
          f"({len(broken_heads)} (l,h)) across selected layers", flush=True)
    return n_heads

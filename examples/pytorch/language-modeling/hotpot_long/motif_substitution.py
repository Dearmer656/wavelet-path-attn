#!/usr/bin/env python3
"""motif_substitution.py — PAT-217 substitution mechanism (self-contained, no modeling edits).

On selected ("broken") heads, replace the RoPE logit with clean NoPE content-QK + a frozen
distilled structural motif b_h(i,j)=s_h(i-j)+v_h(j):
  (a) NoPE: patch rotate_queries_or_keys per selected head so its q/k stay UN-rotated
      (works for both prefill q_len==k_len and KV-cache decode q_len=1).
  (b) motif bias: monkeypatch module-level eager_attention_forward to inject a per-head frozen
      bias via the existing `qwab_bias` additive path (added to pre-softmax logits).

NOTE (PAT-217 review fix): the bias is injected whenever the qwab_bias slot is free, INCLUDING
KV-cache decode (q_len=1, k_len>1) — required so generated-answer attention actually receives
the motif. Pair with run_clm --eval_generate_no_cache True (use_cache=False → q_len==k_len each
step) OR the decode branch below builds the last-row bias.

Control modes: real | nope_only | shuffled_offset | shuffled_sink | shuffled_both |
wrong_head | random_matched.
Tail-extension modes for s(δ>L0): hold_last | median_last64 | zero | exp_decay.
"""
import os
import numpy as np
import torch

_PATCHED = False
_DBG = os.environ.get("MOTIF_DEBUG", "") == "1"
_dbg = {"n": 0, "inj": 0, "seen": 0}
_CACHE_MAX_T = 1024   # cache full [1,nh,T,T] only for small T (training); rebuild+free for eval


def make_masked_rotate(orig_rotate, nope_mask):
    H = int(nope_mask.numel())

    def wrapped(t, *a, **kw):
        rot = orig_rotate(t, *a, **kw)
        if t.dim() == 4 and t.shape[1] == H:
            m = nope_mask.to(device=rot.device, dtype=rot.dtype).view(1, H, 1, 1)
            return rot * (1.0 - m) + t * m
        return rot
    return wrapped


def _extend_s(s_row, L0, T, tail="hold_last"):
    s = np.empty(T, np.float32); s[:L0] = s_row
    if T <= L0:
        return s[:T]
    if tail == "zero":
        s[L0:] = 0.0
    elif tail == "median_last64":
        s[L0:] = np.median(s_row[max(0, L0 - 64):L0])
    elif tail == "exp_decay":
        base = np.median(s_row[max(0, L0 - 64):L0])
        d = np.arange(1, T - L0 + 1)
        s[L0:] = base * np.exp(-d / max(L0, 1))
    else:  # hold_last
        s[L0:] = s_row[L0 - 1]
    return s


def _extend_v(v_row, L0, T):
    v = np.empty(T, np.float32); v[:L0] = v_row
    if T > L0:
        v[L0:] = np.median(v_row[64:L0]) if L0 > 64 else 0.0   # sink lives in early cols
    return v[:T]


def _build_layer_bias(s_lay, v_lay, hmask, lam, L0, T, device, dtype, tail):
    """[1, nh, T, T] additive logit bias; nonzero only on selected heads."""
    nh = s_lay.shape[0]
    off = np.clip(np.arange(T)[:, None] - np.arange(T)[None, :], 0, T - 1)
    B = np.zeros((nh, T, T), dtype=np.float32)
    for h in range(nh):
        if hmask[h] <= 0:
            continue
        s = _extend_s(s_lay[h], L0, T, tail); v = _extend_v(v_lay[h], L0, T)
        B[h] = lam * (s[off] + v[None, :])
    return torch.from_numpy(B).unsqueeze(0).to(device=device, dtype=dtype)


def _install_eager_patch(tail):
    global _PATCHED
    if _PATCHED:
        return
    import transformers.models.gpt2.modeling_gpt2 as M
    _orig = M.eager_attention_forward

    def patched(module, query, key, value, attention_mask, head_mask=None, **kwargs):
        sv = getattr(module, "_motif_sv", None)
        if sv is not None and kwargs.get("qwab_bias", None) is None:
            q_len, k_len = query.size(-2), key.size(-2)
            s_lay, v_lay, hmask, lam, L0 = sv
            dev, dt = query.device, query.dtype
            if _DBG:
                _dbg["seen"] += 1
                inj = (q_len == k_len) or (q_len == 1 and k_len > 1)
                _dbg["inj"] += int(inj)
                if _dbg["n"] < 60:
                    print(f"[MOTIF_DBG] L{getattr(module,'layer_idx','?')} q={q_len} k={k_len} "
                          f"inj={inj}", flush=True); _dbg["n"] += 1
            if q_len == k_len:
                cache = getattr(module, "_motif_bias_cache", {})
                if q_len in cache:
                    b = cache[q_len]
                else:
                    b = _build_layer_bias(s_lay, v_lay, hmask, lam, L0, q_len, dev, dt, tail)
                    if q_len <= _CACHE_MAX_T:            # persist only small T (training)
                        cache[q_len] = b; module._motif_bias_cache = cache
                kwargs["qwab_bias"] = b
            elif q_len == 1 and k_len > 1:               # KV-cache decode: last-row bias only
                full = _build_layer_bias(s_lay, v_lay, hmask, lam, L0, k_len, dev, dt, tail)
                kwargs["qwab_bias"] = full[:, :, k_len - 1:k_len, :]
        return _orig(module, query, key, value, attention_mask, head_mask, **kwargs)

    M.eager_attention_forward = patched
    _PATCHED = True


def select_broken_heads(recon_csv, top_k, npz_path=None, min_train_ve=0.3, nh=12):
    """Rank by OOD recon RMSE, but only among heads whose L512 slash+sink fit is valid
    (train var_explained >= min_train_ve), so we don't inject motifs into heads the
    s(δ)+v(j) model never described."""
    import csv
    rows = []
    with open(recon_csv) as f:
        for r in csv.DictReader(f):
            rows.append((int(r["layer"]), int(r["head"]), float(r["ood_rmse"])))
    assert rows, f"empty recon csv {recon_csv}"
    ve = None
    if npz_path:
        d = np.load(npz_path); ve = d["var_explained"]     # [nl*nh], L512 train fit
    cand = []
    for (l, h, rmse) in rows:
        if ve is None or ve[l * nh + h] >= min_train_ve:
            cand.append((l, h, rmse))
    cand.sort(key=lambda x: -x[2])
    sel = [(l, h) for l, h, _ in cand[:top_k]]
    assert len(sel) == min(top_k, len(cand)), "head selection count mismatch"
    return sel


def _apply_control(S, V, L0, mode, rng):
    if mode == "shuffled_offset":
        for i in range(S.shape[0]):
            S[i] = S[i, rng.permutation(L0)]
    elif mode == "shuffled_sink":
        for i in range(V.shape[0]):
            V[i, :L0] = V[i, rng.permutation(L0)]
    elif mode == "shuffled_both":
        for i in range(S.shape[0]):
            S[i] = S[i, rng.permutation(L0)]; V[i, :L0] = V[i, rng.permutation(L0)]
    elif mode == "wrong_head":                       # derangement (no self-mapping)
        N = S.shape[0]; off = max(1, N // 2)
        perm = (np.arange(N) + off) % N
        assert np.all(perm != np.arange(N)), "derangement has a fixed point"
        S, V = S[perm].copy(), V[perm].copy()
    elif mode == "random_matched":                   # keep global value distribution, kill structure
        sh_s = S.shape; sh_v = V.shape
        S = rng.permutation(S.ravel()).reshape(sh_s).astype(np.float32)
        V = rng.permutation(V.ravel()).reshape(sh_v).astype(np.float32)
    return S, V


def apply_motif_substitution(model, npz_path, broken_heads, lam=1.0, mode="real",
                             nl=12, nh=12, seed=0, tail="hold_last"):
    """mode in {real, nope_only(=none), shuffled_offset, shuffled_sink, shuffled_both,
    wrong_head, random_matched}."""
    import os
    assert os.path.exists(npz_path), f"missing motif npz {npz_path}"
    d = np.load(npz_path)
    S, V = d["s"].astype(np.float32).copy(), d["v"].astype(np.float32).copy()   # [144, L0]
    L0 = int(d["L"])
    assert S.shape[0] == nl * nh, f"motif table rows {S.shape[0]} != {nl*nh}"
    rng = np.random.default_rng(seed)
    use_motif = mode not in ("nope_only", "none")
    if use_motif:
        S, V = _apply_control(S, V, L0, mode, rng)

    per_layer_mask = {l: np.zeros(nh, np.float32) for l in range(nl)}
    for (l, h) in broken_heads:
        per_layer_mask[l][h] = 1.0

    _install_eager_patch(tail)
    import re as _re
    n_heads = 0
    for name, mod in model.named_modules():
        m = _re.search(r"\.h\.(\d+)\.attn$", name)
        if m is None or not hasattr(mod, "rotary_emb"):
            continue
        l = int(m.group(1)); hm = per_layer_mask[l]
        if hm.sum() == 0:
            continue
        if not getattr(mod.rotary_emb, "_motif_patched", False):     # (a) NoPE on selected heads
            mod.rotary_emb.rotate_queries_or_keys = make_masked_rotate(
                mod.rotary_emb.rotate_queries_or_keys, torch.tensor(hm))
            mod.rotary_emb._motif_patched = True
        if use_motif:                                                # (b) stash motif tables
            idx = [l * nh + h for h in range(nh)]
            mod._motif_sv = (S[idx], V[idx], hm, float(lam), L0)
            mod._motif_bias_cache = {}
        n_heads += int(hm.sum())
    print(f"[PAT-217] substitution mode={mode} lam={lam} tail={tail} motif={use_motif} "
          f"on {n_heads} heads ({len(broken_heads)} (l,h))", flush=True)
    assert n_heads == len(broken_heads), "installed head count != requested"
    return n_heads

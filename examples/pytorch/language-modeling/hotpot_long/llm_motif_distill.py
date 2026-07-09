#!/usr/bin/env python3
"""llm_motif_distill.py — PAT-222: distill slash+sink motif from Llama-style LLMs.

Mirrors distill_bias_lowfreq.py (PAT-217) but adapts to:
  - Llama/Mistral attention naming (model.layers[i].self_attn)
  - GQA: num_key_value_heads < num_attention_heads; k repeated via repeat_kv
  - Hooks apply_rotary_pos_emb globally to capture pre/post rotation q, k

Output NPZ has the same schema as distill_bias_lowfreq.py:
  s[nl*nh, T], v[nl*nh, T], var_explained[nl*nh], slash_frac[nl*nh]

Usage:
  python llm_motif_distill.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
      --jsonl data/ruler_eval_2048.jsonl --L 2048 --n_cases 20 \\
      --period_cutoff 2048 --out analysis_outputs/pat222/tinyllama
"""
import argparse, json, os, types
import numpy as np
import torch
import transformers.models.llama.modeling_llama as _M
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# ── per-forward rotation capture ─────────────────────────────────────────────
_CAPTURE = {}   # layer_idx -> {q_pre, q_post, k_pre, k_post}
_LAYER_CTR = [0]

def _install_rotary_capture():
    """Hook Llama-style apply_rotary_pos_emb (full head_dim)."""
    orig = _M.apply_rotary_pos_emb

    def patched(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        li = _LAYER_CTR[0]
        _CAPTURE[li] = {"q_pre": q.detach().float(), "k_pre": k.detach().float()}
        q_r, k_r = orig(q, k, cos, sin, position_ids, unsqueeze_dim)
        _CAPTURE[li]["q_post"] = q_r.detach().float()
        _CAPTURE[li]["k_post"] = k_r.detach().float()
        _LAYER_CTR[0] += 1
        return q_r, k_r

    _M.apply_rotary_pos_emb = patched


def _install_rotary_capture_phi():
    """Hook Phi-style apply_rotary_pos_emb (partial rot_dim only)."""
    import transformers.models.phi.modeling_phi as _PHI
    orig = _PHI.apply_rotary_pos_emb

    def patched(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        li = _LAYER_CTR[0]
        _CAPTURE[li] = {"q_pre": q.detach().float(), "k_pre": k.detach().float()}
        q_r, k_r = orig(q, k, cos, sin, position_ids, unsqueeze_dim)
        _CAPTURE[li]["q_post"] = q_r.detach().float()
        _CAPTURE[li]["k_post"] = k_r.detach().float()
        _LAYER_CTR[0] += 1
        return q_r, k_r

    _PHI.apply_rotary_pos_emb = patched


def _reset_capture():
    _CAPTURE.clear(); _LAYER_CTR[0] = 0


# ── low-freq dim mask (same logic as PAT-208/217) ────────────────────────────
def lowfreq_dim_mask(model, head_dim, period_cutoff):
    """Return float32 mask [head_dim]: 1 on dims whose RoPE period > cutoff."""
    # rotary_emb is at model.model.rotary_emb (shared, new-style Llama) or per-layer attn
    re = getattr(model.model, "rotary_emb", None)
    if re is None:
        re = model.model.layers[0].self_attn.rotary_emb
    inv_freq = re.inv_freq.detach().float().cpu().numpy()   # [head_dim//2]
    periods = 2.0 * np.pi / np.clip(inv_freq, 1e-12, None)
    mask = np.zeros(head_dim, np.float32)
    n_off = 0
    for mi, p in enumerate(periods):
        if p > period_cutoff:
            mask[2 * mi] = 1.0; mask[2 * mi + 1] = 1.0
            n_off += 1
    print(f"[distill] low-freq (P>{period_cutoff}): {int(mask.sum())}/{head_dim} dims "
          f"({n_off}/{len(periods)} pairs)", flush=True)
    return mask


# ── additive fit (same as PAT-217) ───────────────────────────────────────────
def additive_fit(B, ii, jj, off, cnt_off, cnt_col, T, iters=8):
    b = B[ii, jj]
    s = np.zeros(T); v = np.zeros(T)
    for _ in range(iters):
        s = np.bincount(off, weights=(b - v[jj]), minlength=T) / cnt_off
        v = np.bincount(jj, weights=(b - s[off]), minlength=T) / cnt_col
        v -= v.mean()
    recon = s[off] + v[jj]
    ss_res = float(((b - recon) ** 2).sum())
    ss_tot = float(((b - b.mean()) ** 2).sum()) + 1e-9
    slash_e = float((s[off] ** 2).sum())
    sink_e  = float((v[jj] ** 2).sum())
    return s, v, 1.0 - ss_res / ss_tot, slash_e / (slash_e + sink_e + 1e-9)


# ── main ─────────────────────────────────────────────────────────────────────
def run(a):
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(a.model, cache_dir=a.cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    cfg = AutoConfig.from_pretrained(a.model, cache_dir=a.cache_dir)
    cfg._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_pretrained(
        a.model, config=cfg, torch_dtype=torch.float32, cache_dir=a.cache_dir,
        trust_remote_code=True).eval().to(dev)

    nl  = cfg.num_hidden_layers
    nh  = cfg.num_attention_heads
    nkv = cfg.num_key_value_heads
    g   = nh // nkv          # KV groups (8 for TinyLlama, 1 for Phi-2)
    hd  = cfg.hidden_size // nh
    T   = a.L
    scale = 1.0 / np.sqrt(hd)

    is_phi = getattr(cfg, "model_type", "") == "phi"

    dmask_np = lowfreq_dim_mask(model, hd, a.period_cutoff)
    dm = torch.tensor(dmask_np, device=dev).view(1, 1, hd)   # [1,1,hd]

    if is_phi:
        _install_rotary_capture_phi()
    else:
        _install_rotary_capture()

    # load cases
    cases = []
    with open(a.jsonl) as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            if int(r.get("length", r.get("meta", {}).get("target_total_tokens", -1))) != a.L:
                continue
            inp = r.get("input", "")
            ids = tok(inp, add_special_tokens=True, truncation=False)["input_ids"]
            if len(ids) >= a.L:
                cases.append(ids[:a.L])
            if len(cases) >= a.n_cases:
                break
    print(f"[distill] L={a.L}: {len(cases)} cases loaded", flush=True)
    assert cases, "no cases found — check --jsonl and --L"

    # accumulate low-freq residual
    Bsum = np.zeros((nl, nh, T, T), np.float64)

    for ci, ids in enumerate(cases):
        _reset_capture()
        with torch.no_grad():
            model(torch.tensor([ids], device=dev))

        for li in range(nl):
            cap = _CAPTURE[li]
            q_pre  = cap["q_pre"][0]   # [nh,  T, rot_dim]  (rot_dim==hd for Llama; <hd for Phi)
            q_post = cap["q_post"][0]
            k_pre  = cap["k_pre"][0]   # [nkv, T, rot_dim]
            k_post = cap["k_post"][0]

            # repeat KV to match Q heads (no-op for Phi-2 which has no GQA)
            k_pre  = k_pre.repeat_interleave(g, dim=0)
            k_post = k_post.repeat_interleave(g, dim=0)

            # dm_dev sliced to match the actual rotated dim (Phi-2: rot_dim=32, hd=80)
            rot_dim = q_pre.shape[-1]
            dm_dev  = dm.squeeze()[:rot_dim].to(q_post.device)   # [rot_dim]

            qp = (q_post * dm_dev); kp = (k_post * dm_dev)   # low-freq post-RoPE
            qn = (q_pre  * dm_dev); kn = (k_pre  * dm_dev)   # low-freq pre-RoPE (NoPE)

            B_rope = torch.einsum("htd,hsd->hts", qp, kp).cpu().numpy() * scale
            B_nope = torch.einsum("htd,hsd->hts", qn, kn).cpu().numpy() * scale
            Bsum[li] += (B_rope - B_nope)

        print(f"  case {ci} done", flush=True)

    B = (Bsum / len(cases)).astype(np.float32)   # [nl, nh, T, T]

    tril = np.tril_indices(T); ii, jj = tril; off = ii - jj
    cnt_off = np.maximum(np.bincount(off, minlength=T), 1)
    cnt_col = np.maximum(np.bincount(jj,  minlength=T), 1)

    S_tab = np.zeros((nl * nh, T), np.float32)
    V_tab = np.zeros((nl * nh, T), np.float32)
    rows  = []
    for li in range(nl):
        for hi in range(nh):
            s, v, ve, sf = additive_fit(B[li, hi], ii, jj, off, cnt_off, cnt_col, T)
            idx = li * nh + hi
            S_tab[idx] = s; V_tab[idx] = v
            rows.append({"layer": li, "head": hi, "var_explained": ve, "slash_frac": sf})
            if hi == 0:
                print(f"  L{li:02d} h00 var_exp={ve:.4f} slash_frac={sf:.3f}", flush=True)

    npz_path = os.path.join(a.out, f"llm_motif_L{a.L}.npz")
    np.savez(npz_path, s=S_tab, v=V_tab,
             var_explained=np.array([r["var_explained"] for r in rows]),
             slash_frac=np.array([r["slash_frac"] for r in rows]),
             nl=nl, nh=nh, L0=T, period_cutoff=a.period_cutoff,
             dmask=dmask_np)
    print(f"[distill] saved → {npz_path}", flush=True)

    import csv
    csv_path = os.path.join(a.out, f"llm_motif_L{a.L}_stats.csv")
    with open(csv_path, "w", newline="") as cf:
        w = csv.DictWriter(cf, fieldnames=["layer", "head", "var_explained", "slash_frac"])
        w.writeheader(); w.writerows(rows)
    print(f"[distill] stats → {csv_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",         default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--jsonl",         required=True)
    ap.add_argument("--L",             type=int, default=2048)
    ap.add_argument("--n_cases",       type=int, default=20)
    ap.add_argument("--period_cutoff", type=float, default=2048.0)
    ap.add_argument("--out",           default="analysis_outputs/pat222/tinyllama")
    ap.add_argument("--cache_dir",     default="/cl/work5/hongyu-s/huggingfac")
    run(ap.parse_args())

#!/usr/bin/env python3
"""
PAT-225 Direction E: what does QWAB's router actually read from PaTH's
accumulated Householder state q_corr, and does it convert it into a
position-conditioned routing signal?

For feat_mode=q_minus_qcorr_meanh the router input is feat_ln((qf-q_corr)
.mean_over_heads) — i.e. the LayerNorm (direction-only) of delta = qf-q_corr,
where q_corr = (prod H_t) qf is PaTH's cumulative-state query. So the router's
entire signal is "how much / in what direction PaTH's Householder accumulation
moved the query." This probe measures, on real HotpotQA-Long L4096 inputs:

  (1) rel_delta = ||qf-q_corr|| / ||qf||  vs position  -> does PaTH's state
      transform the query MORE at later positions (the accumulation signature)?
  (2) corr(position, rel_delta)                        -> is that growth real?
  (3) std of pi_null / router entropy across positions -> does the router
      actually vary its routing with position, or is it effectively static
      (which would mean it is NOT reading q_corr in a useful way)?

Run on the best kaiming s42 ckpt-15000 (F1=0.7038), plus rerun and zeroinit
for contrast. Single GPU, ~15 min.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/project/nlp-work5/hongyu-s/transformers/src")
sys.path.insert(0, "/project/nlp-work5/hongyu-s/flash-linear-attention")

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import fla.layers.path_attn as pa

HOTPOT_JSONL = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/hotpot_long_dev_uniform.jsonl"
N_SAMPLES = 12
TARGET_LEN = 4096
N_LAYERS = 12
N_POS_BUCKETS = 16

BASE = "/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat225_scale_card"
RUNS = {
    "orig_s42": f"{BASE}/S4_s42/checkpoint-15000",
    "rerun_s42": f"{BASE}/S4_s42_rerun/checkpoint-15000",
    "zeroinit_s42": f"{BASE}/S4_s42_zeroinit/checkpoint-15000",
}


def render_input(question, context):
    ctx = "".join(f"Title: {t}\n{' '.join(s).strip()}\n\n" for t, s in context)
    return f"Context:\n{ctx}\nQuestion: {question}\nAnswer:"


def load_examples():
    exs = []
    with open(HOTPOT_JSONL) as f:
        for line in f:
            ex = json.loads(line)
            if ex["meta"]["target_total_tokens"] == TARGET_LEN:
                exs.append(ex)
            if len(exs) >= N_SAMPLES:
                break
    return exs


# ---- monkeypatch router feature to stash qf / q_corr per layer ----
_ORIG_FEAT = pa.PaTHAttention._ctxscale_router_feature


def _patched_feat(self, qf, q_corr, *, use_mlp=False, hidden_states=None):
    try:
        self._probe_qf = qf.detach().float()
        self._probe_qcorr = q_corr.detach().float()
    except Exception:
        pass
    return _ORIG_FEAT(self, qf, q_corr, use_mlp=use_mlp, hidden_states=hidden_states)


pa.PaTHAttention._ctxscale_router_feature = _patched_feat


def load_model(ckpt, device):
    cfg = AutoConfig.from_pretrained(ckpt)
    cfg.attn_implementation = "path_attn"
    cfg.wavelet_logit_bias_log_every = 10**9
    cfg.wavelet_ctxscale_eval_log_once = False
    m = AutoModelForCausalLM.from_pretrained(ckpt, config=cfg, torch_dtype=torch.float32).to(device)
    m.eval()
    return m


def router_pi_hook(store):
    def hook(module, inp, out, layer_idx):
        logits = out.detach().float()
        g = torch.sigmoid(logits[..., 1:])
        w = g / g.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        g0 = torch.sigmoid(logits[..., 0:1])
        pi_scale = g0 * w
        pi_null = (1.0 - g0)
        pi = torch.cat([pi_null, pi_scale], dim=-1)  # [...,K+1]
        if pi.dim() == 4:
            pi = pi.mean(dim=2)  # mean over heads -> [B,T,K+1]
        store[layer_idx] = pi.squeeze(0).cpu().numpy()  # [T,K+1]
    return hook


def bucketize(vec_t, n_buckets):
    T = len(vec_t)
    idx = (np.linspace(0, n_buckets, T, endpoint=False)).astype(int)
    idx = np.clip(idx, 0, n_buckets - 1)
    out = np.full(n_buckets, np.nan)
    for b in range(n_buckets):
        m = idx == b
        if m.any():
            out[b] = np.nanmean(vec_t[m])
    return out


def run_one(ckpt, examples, tokenizer, device):
    model = load_model(ckpt, device)
    pi_store = {}
    handles = []
    modules_by_layer = {}
    for name, mod in model.named_modules():
        if name.endswith("wavelet_ctx_router") and isinstance(mod, torch.nn.Linear):
            parts = name.split(".")
            L = None
            for i, p in enumerate(parts):
                if p == "h" and i + 1 < len(parts):
                    L = int(parts[i + 1])
            if L is not None:
                handles.append(mod.register_forward_hook(
                    lambda m, i, o, LL=L: router_pi_hook(pi_store)(m, i, o, LL)))
        # find the PaTHAttention core module per layer (path: ...h.{L}.attn.core)
        if isinstance(mod, pa.PaTHAttention):
            parts = name.split(".")
            L = None
            for i, p in enumerate(parts):
                if p == "h" and i + 1 < len(parts):
                    L = int(parts[i + 1])
            if L is not None:
                modules_by_layer[L] = mod

    # accumulators
    rel_curve = {L: [] for L in range(N_LAYERS)}   # per-example bucketed rel_delta
    pinull_curve = {L: [] for L in range(N_LAYERS)}
    ent_curve = {L: [] for L in range(N_LAYERS)}
    rel_pos_corr = {L: [] for L in range(N_LAYERS)}
    pinull_pos_std = {L: [] for L in range(N_LAYERS)}
    ent_pos_std = {L: [] for L in range(N_LAYERS)}

    for ex in examples:
        pi_store.clear()
        prompt = render_input(ex["question"], ex["context"])
        enc = tokenizer(prompt, return_tensors="pt", truncation=False)
        input_ids = enc["input_ids"].to(device)
        with torch.no_grad():
            model(input_ids=input_ids)
        for L in range(N_LAYERS):
            mod = modules_by_layer.get(L)
            if mod is None or not hasattr(mod, "_probe_qf"):
                continue
            qf = mod._probe_qf.squeeze(0)      # [T,H,d]
            qc = mod._probe_qcorr.squeeze(0)   # [T,H,d]
            delta = qf - qc
            qf_norm = qf.norm(dim=-1).mean(dim=-1)      # [T] mean over heads
            d_norm = delta.norm(dim=-1).mean(dim=-1)    # [T]
            rel = (d_norm / qf_norm.clamp_min(1e-8)).cpu().numpy()  # [T]
            T = len(rel)
            pos = np.arange(T)
            rel_curve[L].append(bucketize(rel, N_POS_BUCKETS))
            if T > 2:
                rel_pos_corr[L].append(float(np.corrcoef(pos, rel)[0, 1]))
            if L in pi_store:
                pi = pi_store[L]  # [T,K+1]
                pinull = pi[:, 0]
                ps = pi[:, 1:]
                ent = -(ps * np.log(ps + 1e-9)).sum(axis=1)  # scale entropy per position
                pinull_curve[L].append(bucketize(pinull, N_POS_BUCKETS))
                ent_curve[L].append(bucketize(ent, N_POS_BUCKETS))
                pinull_pos_std[L].append(float(np.std(pinull)))
                ent_pos_std[L].append(float(np.std(ent)))

    for h in handles:
        h.remove()
    del model
    torch.cuda.empty_cache()

    def agg(d):
        return {L: (np.nanmean(np.stack(v), axis=0).tolist() if v else None) for L, v in d.items()}

    def aggs(d):
        return {L: (float(np.nanmean(v)) if v else None) for L, v in d.items()}

    return {
        "rel_curve": agg(rel_curve),
        "pinull_curve": agg(pinull_curve),
        "ent_curve": agg(ent_curve),
        "rel_pos_corr": aggs(rel_pos_corr),
        "pinull_pos_std": aggs(pinull_pos_std),
        "ent_pos_std": aggs(ent_pos_std),
    }


def main():
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    examples = load_examples()
    print(f"Loaded {len(examples)} L{TARGET_LEN} examples", flush=True)

    results = {}
    for run, ckpt in RUNS.items():
        if not Path(ckpt).is_dir():
            print(f"SKIP {run}: {ckpt} missing", flush=True)
            continue
        print(f"=== {run} ===", flush=True)
        results[run] = run_one(ckpt, examples, tokenizer, device)

    out = Path(__file__).parent / "results" / "pat225_E_qcorr_router.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, "w"), indent=2)
    print(f"Saved {out}")

    print("\n=== SUMMARY: does PaTH-state delta grow with position, and does routing vary with it? ===")
    for run in results:
        r = results[run]
        print(f"\n--- {run} ---")
        print("  L  | mean_rel_delta | corr(pos,rel) | pi_null_pos_std | ent_pos_std")
        for L in range(N_LAYERS):
            rc = r["rel_curve"].get(L) or r["rel_curve"].get(str(L))
            mean_rel = float(np.nanmean(rc)) if rc else float("nan")
            c = r["rel_pos_corr"].get(L) or r["rel_pos_corr"].get(str(L))
            ns = r["pinull_pos_std"].get(L) or r["pinull_pos_std"].get(str(L))
            es = r["ent_pos_std"].get(L) or r["ent_pos_std"].get(str(L))
            print(f"  {L:>2} | {mean_rel:>13.4f} | {(c if c is not None else float('nan')):>13.4f} "
                  f"| {(ns if ns is not None else float('nan')):>15.5f} | {(es if es is not None else float('nan')):>10.5f}")


if __name__ == "__main__":
    main()

"""
Compare positional distinguishability of the QWAB router's conditioning feature
between a NoPE+QWAB checkpoint and the canonical PaTH+QWAB checkpoint.

Hypothesis under test: NoPE hidden states carry no positional information, so
query positions become increasingly indistinguishable (high cosine similarity)
beyond the training length, while PaTH's q-q_corr conditioning stays more
position-differentiated. This directly tests the (otherwise unverified) claim
that the NoPE router "learns a fixed, length-invariant bias that... actively
corrupts attention once extrapolated."

N=100 examples, all 12 layers. For each (example, layer) we compute the mean
off-diagonal cosine similarity of the router-input feature within two position
regions: in-distribution (<= L_train) and extrapolation (> L_train). We then
aggregate across examples and layers (mean +/- std), and also dump a few
layer-averaged similarity matrices for heatmap figures.

Usage: python query_similarity_nope_vs_path.py
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling")
sys.path.insert(0, "/project/nlp-work5/hongyu-s/transformers/src")
sys.path.insert(0, "/project/nlp-work5/hongyu-s/flash-linear-attention")

import run_clm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

L_TRAIN = 512
L_EVAL = 2048
N_EXAMPLES = 100
HEATMAP_EXAMPLES = 3  # dump layer-averaged sim matrices for the first few examples
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/analysis/query_similarity_out"

NOPE_CKPT = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/small_nope_qwab_10ep_s42/checkpoint-15000"
PATH_CKPT = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat244_dual_temp/K1_L512_me16_rho256_ricker_s42/checkpoint-15000"
PATH_CFG = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat244_dual_temp/K1_L512_me16_rho256_ricker_s42/supply_model.cfg"
JSONL = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/hotpot_long_dev_uniform.jsonl"


def load_long_examples(tokenizer, target_len, n):
    out = []
    with open(JSONL) as f:
        for line in f:
            ex = json.loads(line)
            ctx = ex.get("context")
            if not ctx:
                continue
            text = " ".join(sent for _title, sents in ctx for sent in sents)
            ids = tokenizer(text, return_tensors="pt")["input_ids"]
            if ids.shape[1] >= target_len:
                out.append(ids[:, :target_len])
            if len(out) >= n:
                break
    if len(out) < n:
        print(f"[warn] only found {len(out)} examples >= {target_len} tokens (wanted {n})")
    return out


def load_nope_model():
    tok = AutoTokenizer.from_pretrained(NOPE_CKPT)
    config = AutoConfig.from_pretrained(NOPE_CKPT)
    # May-2026 checkpoint: K=8 (router_band_num) but scale_max_exp=None -> single
    # 16.0, rejected by the current QWABBias.__init__. We only capture the router
    # INPUT hidden_states (pre-scale), so the scale grid is irrelevant; inject a
    # length-K placeholder just to pass validation. Router weight shape depends
    # only on K, so the saved weights still load.
    k = int(getattr(config, "wavelet_ctxscale_k", None) or getattr(config, "router_band_num", 8))
    sme = getattr(config, "wavelet_ctxscale_scale_max_exp", None)
    if not isinstance(sme, (list, tuple)) or len(sme) != k:
        config.wavelet_ctxscale_scale_max_exp = [16.0] * k
    config.wavelet_ctxscale_k = k
    model = AutoModelForCausalLM.from_pretrained(NOPE_CKPT, config=config, attn_implementation="eager")
    model.to(DEVICE).eval()
    return model, tok


def load_path_model():
    tok = AutoTokenizer.from_pretrained("gpt2")
    config = AutoConfig.from_pretrained(PATH_CKPT)
    cfg = run_clm.read_kv_config(PATH_CFG)
    run_clm.add_missing_to_hf_config(config, cfg)
    config.attn_implementation = "path_attn"
    model = AutoModelForCausalLM.from_pretrained(PATH_CKPT, config=config)
    model.to(DEVICE).eval()
    return model, tok


def region_means(feat):
    """feat: [T, D] router-input feature for one (example, layer).
    Returns (in_dist_mean, extrap_mean) off-diagonal cosine similarities."""
    feat = torch.nn.functional.normalize(feat.float(), dim=-1)
    T = feat.shape[0]

    def block_mean(lo, hi):
        n = hi - lo
        if n < 2:
            return float("nan")
        sim = feat[lo:hi] @ feat[lo:hi].T
        off = ~torch.eye(n, dtype=torch.bool, device=sim.device)
        return sim[off].mean().item()

    in_dist = block_mean(0, min(L_TRAIN, T))
    extrap = block_mean(L_TRAIN, T) if T > L_TRAIN else float("nan")
    return in_dist, extrap


def capture_nope(model, input_ids):
    from transformers.models.gpt2.modeling_gpt2 import QWABBias

    captured = {}
    orig = QWABBias.forward

    def patched(self, hidden_states, chunk_size=None):
        captured[getattr(self, "_cap_id", None)] = hidden_states.detach()[0].cpu()
        return orig(self, hidden_states, chunk_size=chunk_size)

    QWABBias.forward = patched
    for i, block in enumerate(model.transformer.h):
        mod = getattr(block, "attn", None)
        if mod is not None and getattr(mod, "qwab_bias_module", None) is not None:
            mod.qwab_bias_module._cap_id = i
    try:
        with torch.no_grad():
            model(input_ids=input_ids.to(DEVICE))
    finally:
        QWABBias.forward = orig
    return captured


def capture_path(model, input_ids):
    from fla.layers.path_attn import PaTHAttention

    captured = {}
    orig = PaTHAttention._ctxscale_router_feature

    def patched(self, qf, q_corr, *, use_mlp=False, hidden_states=None):
        out = orig(self, qf, q_corr, use_mlp=use_mlp, hidden_states=hidden_states)
        captured[int(self.layer_idx)] = out.detach()[0].cpu()
        return out

    PaTHAttention._ctxscale_router_feature = patched
    try:
        with torch.no_grad():
            model(input_ids=input_ids.to(DEVICE))
    finally:
        PaTHAttention._ctxscale_router_feature = orig
    return captured


def run_model(label, load_fn, capture_fn, tokenizer_for_examples=None):
    model, tok = load_fn()
    examples = load_long_examples(tok, L_EVAL, N_EXAMPLES)
    # per-layer lists of (in_dist, extrap) across examples
    per_layer_in = {}
    per_layer_extrap = {}
    heatmap_acc = None  # [n_layers, T, T] running sum for first HEATMAP_EXAMPLES
    heatmap_count = 0
    for ei, ids in enumerate(examples):
        feats = capture_fn(model, ids)
        if not feats:
            print(f"[{label}] ex{ei}: no features captured")
            continue
        layers = sorted(feats.keys())
        for lid in layers:
            ind, ext = region_means(feats[lid])
            per_layer_in.setdefault(lid, []).append(ind)
            per_layer_extrap.setdefault(lid, []).append(ext)
        if ei < HEATMAP_EXAMPLES:
            mats = []
            for lid in layers:
                f = torch.nn.functional.normalize(feats[lid].float(), dim=-1)
                mats.append((f @ f.T))
            stack = torch.stack(mats, 0)  # [L, T, T]
            heatmap_acc = stack if heatmap_acc is None else heatmap_acc + stack
            heatmap_count += 1
        if (ei + 1) % 20 == 0:
            print(f"[{label}] processed {ei + 1}/{len(examples)}")
    del model
    torch.cuda.empty_cache()

    # aggregate across all (example, layer) pairs
    all_in = np.array([v for lst in per_layer_in.values() for v in lst], dtype=np.float64)
    all_ext = np.array([v for lst in per_layer_extrap.values() for v in lst], dtype=np.float64)
    all_in = all_in[~np.isnan(all_in)]
    all_ext = all_ext[~np.isnan(all_ext)]
    result = {
        "in_mean": float(all_in.mean()), "in_std": float(all_in.std()),
        "extrap_mean": float(all_ext.mean()), "extrap_std": float(all_ext.std()),
        "n_pairs": int(all_in.size),
        "per_layer_in_mean": {int(k): float(np.nanmean(v)) for k, v in per_layer_in.items()},
        "per_layer_extrap_mean": {int(k): float(np.nanmean(v)) for k, v in per_layer_extrap.items()},
    }
    if heatmap_acc is not None and heatmap_count > 0:
        avg = (heatmap_acc / heatmap_count).mean(0).numpy()  # layer-averaged, [T, T]
        os.makedirs(OUT_DIR, exist_ok=True)
        np.save(os.path.join(OUT_DIR, f"heatmap_{label}.npy"), avg.astype(np.float32))
    return result


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"=== NoPE+QWAB (N={N_EXAMPLES}) ===")
    r_nope = run_model("NoPE", load_nope_model, capture_nope)
    print(f"=== PaTH+QWAB canonical (N={N_EXAMPLES}) ===")
    r_path = run_model("PaTH", load_path_model, capture_path)

    print("\n===== RESULTS (mean off-diagonal cosine similarity of router-input feature) =====")
    for label, r in (("PaTH (q-q_corr)", r_path), ("NoPE (hidden state)", r_nope)):
        print(f"{label}:")
        print(f"  in-distribution (<= {L_TRAIN}): {r['in_mean']:.4f} +/- {r['in_std']:.4f}")
        print(f"  extrapolation   (> {L_TRAIN}):  {r['extrap_mean']:.4f} +/- {r['extrap_std']:.4f}")
        print(f"  n (example x layer) pairs: {r['n_pairs']}")
    print("\nDelta extrap (NoPE - PaTH): "
          f"{r_nope['extrap_mean'] - r_path['extrap_mean']:+.4f}")
    print("Within-model extrap - in_dist rise:")
    print(f"  NoPE: {r_nope['extrap_mean'] - r_nope['in_mean']:+.4f}")
    print(f"  PaTH: {r_path['extrap_mean'] - r_path['in_mean']:+.4f}")

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump({"NoPE": r_nope, "PaTH": r_path,
                   "L_train": L_TRAIN, "L_eval": L_EVAL, "n_examples": N_EXAMPLES}, f, indent=2)
    print(f"\nsaved: {OUT_DIR}/summary.json  +  heatmap_{{NoPE,PaTH}}.npy")


if __name__ == "__main__":
    main()

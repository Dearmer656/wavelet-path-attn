"""
Compare positional distinguishability of the QWAB router's conditioning feature
between a NoPE+QWAB checkpoint and the canonical PaTH+QWAB checkpoint.

Hypothesis under test: NoPE hidden states carry no positional information, so
query positions become increasingly indistinguishable (high cosine similarity)
beyond the training length, while PaTH's q-q_corr conditioning stays more
position-differentiated. This would support the claim (currently unverified)
that "the [NoPE] router learns a fixed, length-invariant bias that... actively
corrupts attention once extrapolated."

Usage: python query_similarity_nope_vs_path.py
"""
import json
import sys

import torch

sys.path.insert(0, "/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling")
sys.path.insert(0, "/project/nlp-work5/hongyu-s/transformers/src")
sys.path.insert(0, "/project/nlp-work5/hongyu-s/flash-linear-attention")

import run_clm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

L_TRAIN = 512
L_EVAL = 2048
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NOPE_CKPT = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/small_nope_qwab_10ep_s42/checkpoint-15000"
PATH_CKPT = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat244_dual_temp/K1_L512_me16_rho256_ricker_s42/checkpoint-15000"
PATH_CFG = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat244_dual_temp/K1_L512_me16_rho256_ricker_s42/supply_model.cfg"
JSONL = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/hotpot_long_dev_uniform.jsonl"


def load_long_example(tokenizer, target_len):
    with open(JSONL) as f:
        for line in f:
            ex = json.loads(line)
            ctx = ex.get("context")
            if not ctx:
                continue
            text = " ".join(
                sent for _title, sents in ctx for sent in sents
            )
            ids = tokenizer(text, return_tensors="pt")["input_ids"]
            if ids.shape[1] >= target_len:
                return ids[:, :target_len]
    raise RuntimeError(f"No example >= {target_len} tokens found in {JSONL}")


def load_nope_model():
    tok = AutoTokenizer.from_pretrained(NOPE_CKPT)
    model = AutoModelForCausalLM.from_pretrained(NOPE_CKPT, attn_implementation="eager")
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


def capture_nope_features(model, input_ids):
    from transformers.models.gpt2.modeling_gpt2 import QWABBias

    captured = {}
    orig_forward = QWABBias.forward

    def patched(self, hidden_states, chunk_size=None):
        layer_id = getattr(self, "_capture_layer_id", None)
        captured.setdefault(layer_id, hidden_states.detach().clone())
        return orig_forward(self, hidden_states, chunk_size=chunk_size)

    QWABBias.forward = patched
    for i, block in enumerate(model.transformer.h):
        for name in ("attn", "self_attn"):
            mod = getattr(block, name, None)
            if mod is not None and getattr(mod, "qwab_bias_module", None) is not None:
                mod.qwab_bias_module._capture_layer_id = i
    try:
        with torch.no_grad():
            model(input_ids=input_ids.to(DEVICE))
    finally:
        QWABBias.forward = orig_forward
    return captured


def capture_path_features(model, input_ids):
    from fla.layers.path_attn import PaTHAttention

    captured = {}
    orig_fn = PaTHAttention._ctxscale_router_feature

    def patched(self, qf, q_corr, *, use_mlp=False, hidden_states=None):
        out = orig_fn(self, qf, q_corr, use_mlp=use_mlp, hidden_states=hidden_states)
        captured.setdefault(int(self.layer_idx), out.detach().clone())
        return out

    PaTHAttention._ctxscale_router_feature = patched
    try:
        with torch.no_grad():
            model(input_ids=input_ids.to(DEVICE))
    finally:
        PaTHAttention._ctxscale_router_feature = orig_fn
    return captured


def similarity_report(captured, label):
    if not captured:
        print(f"[{label}] no features captured -- check hook wiring")
        return
    layers = sorted(captured.keys())
    mid_layer = layers[len(layers) // 2]
    feat = captured[mid_layer][0]  # [T, D], batch=0
    feat = torch.nn.functional.normalize(feat.float(), dim=-1)
    sim = feat @ feat.T  # [T, T]
    T = sim.shape[0]
    mask_eye = ~torch.eye(T, dtype=torch.bool)

    def region_mean(lo, hi):
        block = sim[lo:hi, lo:hi]
        m = ~torch.eye(hi - lo, dtype=torch.bool)
        return block[m].mean().item()

    in_dist = region_mean(0, min(L_TRAIN, T))
    if T > L_TRAIN:
        extrap = region_mean(L_TRAIN, T)
    else:
        extrap = float("nan")
    overall = sim[mask_eye].mean().item()
    print(f"[{label}] layer={mid_layer} T={T}")
    print(f"  in-distribution (pos 0-{L_TRAIN}) mean pairwise cos-sim: {in_dist:.4f}")
    print(f"  extrapolation   (pos {L_TRAIN}-{T}) mean pairwise cos-sim: {extrap:.4f}")
    print(f"  overall mean pairwise cos-sim: {overall:.4f}")
    return {"in_dist": in_dist, "extrap": extrap, "overall": overall}


def main():
    print("=== Loading NoPE+QWAB ===")
    nope_model, nope_tok = load_nope_model()
    ids_nope = load_long_example(nope_tok, L_EVAL)
    nope_feats = capture_nope_features(nope_model, ids_nope)
    del nope_model
    torch.cuda.empty_cache()

    print("=== Loading PaTH+QWAB (canonical) ===")
    path_model, path_tok = load_path_model()
    ids_path = load_long_example(path_tok, L_EVAL)
    path_feats = capture_path_features(path_model, ids_path)
    del path_model
    torch.cuda.empty_cache()

    print("\n=== Results ===")
    r_nope = similarity_report(nope_feats, "NoPE")
    r_path = similarity_report(path_feats, "PaTH")

    if r_nope and r_path:
        print("\n=== Delta (NoPE - PaTH) ===")
        for k in ("in_dist", "extrap", "overall"):
            print(f"  {k}: {r_nope[k] - r_path[k]:+.4f}")


if __name__ == "__main__":
    main()

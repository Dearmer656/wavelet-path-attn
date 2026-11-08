#!/usr/bin/env python3
"""PAT-310: PaTH content vs transport decomposition at long L (GPT-2 medium, up to L16384).

Same statistics as probe_path_transport_far_near.py (stats_one / reduce), but without the
[H,T,T] UT-form matrices, which do not fit at L16384. Only the last Q_ROWS answer-side
query rows are needed, so each row's PaTH logit is computed exactly by walking back
through the Householder transitions:
  E[i,j] = (H_{j+1} ... H_i q_i) . k_j,   H_t = I - beta_t w_t w_t^T
  C[i,j] = q_i . k_j,   R = C - E
The forward pass runs on the Triton kernel; q, k, w, beta are captured at its input.
At L >= 8192 this is also the kernel used for the paper's evaluation (no QWAB bias).

--validate L: run the model once on the PyTorch kernel (capturing C, R from
path_ut_base_raw) and once on Triton (recurrence), and report the max deviation.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_path_transport_far_near as P
pa = P.pa
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

R = Path("/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs")
DATA = Path("/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data")
HOTPOT = {16384: DATA / "hotpot_long_dev_uniform_16384_large_pool.jsonl"}  # the L16384 eval file
MODELS = {
    "PaTH_s42": R / "mix_medium_owt_PA_10ep",  # paper Table 8 PaTH seeds
    "PaTH_s43": R / "mix_medium_PA_only_s43",
    "PaTH_s44": R / "mix_medium_PA_only_s44",
    "QWAB_s42": R / "mix_medium_K1_rho256_ricker_s42",
    "QWAB_s43": R / "mix_medium_K1_rho256_ricker_s43",
    "QWAB_s44": R / "mix_medium_K1_rho256_ricker_s44",
}
P.BINS = [1, 4, 16, 64, 256, 512, 1024, 2048, 4096, 8192, 16384]
Q = P.Q_ROWS


class TritonCapture:
    """Wraps parallel_path_attn and keeps the last-Q query rows plus full k, w, beta per layer."""
    def __init__(self):
        self.layers = []

    def install(self):
        self.orig = pa.parallel_path_attn
        cap = self

        def wrapped(q, k, v, w, beta, g=None, cu_seqlens=None, **kw):
            if g is not None: raise RuntimeError("forget gate present; probe assumes g=None")
            cap.layers.append((q[0, -Q:].float().clone(), k[0].float().clone(), w[0].float().clone(), beta[0].float().clone()))
            return cap.orig(q=q, k=k, v=v, w=w, beta=beta, g=g, cu_seqlens=cu_seqlens, **kw)

        pa.parallel_path_attn = wrapped

    def uninstall(self):
        pa.parallel_path_attn = self.orig


@torch.no_grad()
def rows_by_recurrence(layers):
    """layers: list of (q_rows [Q,H,d], k [T,H,d], w [T,H,d], beta [T,H]). Returns C, E: [Ly,H,Q,T]."""
    qr = torch.stack([x[0] for x in layers]).double()   # [Ly,Q,H,d]
    K = torch.stack([x[1] for x in layers]).double()    # [Ly,T,H,d]
    W = torch.stack([x[2] for x in layers]).double()
    Bt = torch.stack([x[3] for x in layers]).double()   # [Ly,T,H]
    Ly, T, H, d = K.shape
    C = torch.einsum("lqhd,lthd->lhqt", qr, K)
    U = torch.zeros(Ly, H, Q, d, dtype=torch.float64, device=K.device)
    E = torch.zeros(Ly, H, Q, T, dtype=torch.float64, device=K.device)
    for t in range(T - 1, -1, -1):
        r = t - (T - Q)
        if r >= 0: U[:, :, r] = qr[:, r]
        E[..., t] = torch.einsum("lhqd,lhd->lhq", U, K[:, t])
        wt = W[:, t]                                       # [Ly,H,d]
        dot = torch.einsum("lhqd,lhd->lhq", U, wt) * Bt[:, t].unsqueeze(-1)
        U -= dot.unsqueeze(-1) * wt.unsqueeze(2)
    return C.float(), E.float()


def load_model(ck, impl, dev):
    cfg = AutoConfig.from_pretrained(ck); P.apply_supply_cfg(cfg, ck.parent)
    cfg.attn_implementation = "path_attn"; cfg.path_attn_impl = impl
    cfg.wavelet_logit_bias_log_every = 10 ** 9; cfg.wavelet_ctxscale_eval_log_once = False
    model = AutoModelForCausalLM.from_pretrained(ck, config=cfg, torch_dtype=torch.float32).to(dev).eval()
    for m in model.modules():
        if isinstance(m, pa.PaTHAttention): m._capture_debug_tensors = False
    return cfg, model


def examples(path, L, n):
    out = []
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            if ex["meta"]["target_total_tokens"] == L: out.append(ex)
            if len(out) >= n: break
    return out


def validate(name, L, dev, tok):
    ck = MODELS[name] / "checkpoint-15000"
    ex = examples(P.HOTPOT, L, 1)[0]
    ids, _, _ = P.render_with_support(ex, tok)
    cfg, model = load_model(ck, "pytorch", dev)
    cap = P.Capture(); cap.install()
    try:
        with torch.no_grad(): model(input_ids=ids.to(dev))
    finally: cap.uninstall()
    ref = {l: cap.base[l] for l in cap.base}
    del model; torch.cuda.empty_cache()
    cfg, model = load_model(ck, "triton", dev)
    tc = TritonCapture(); tc.install()
    try:
        with torch.no_grad(): model(input_ids=ids.to(dev))
    finally: tc.uninstall()
    C, E = rows_by_recurrence(tc.layers)
    T = ids.shape[1]
    i = torch.arange(T - Q, T, device=dev).view(Q, 1); j = torch.arange(T, device=dev).view(1, T)
    valid = (j <= i)
    for l in sorted(ref):
        Cr, Rr = ref[l]; Er = Cr - Rr
        Rn = C[l] - E[l]
        dC = float(((C[l] - Cr).abs() * valid).max()); dR = float(((Rn - Rr).abs() * valid).max())
        sR = float((Rr.abs() * valid).max())
        print(f"[validate] {name} L{T} layer {l}: max|dC|={dC:.2e} max|dR|={dR:.2e} (max|R|={sR:.2e})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--L", type=int, default=16384)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--validate", type=int, default=0)
    ap.add_argument("--out", default=str(R / "pat310_c2/transport_probe_medium"))
    a = ap.parse_args()
    dev = torch.device("cuda"); tok = AutoTokenizer.from_pretrained("gpt2")
    if a.validate:
        validate(a.models.split(",")[0], a.validate, dev, tok); return
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for name in a.models.split(","):
        f = out / f"{name}_L{a.L}.json"
        if f.exists(): print(f"[have] {f.name}"); continue
        cfg, model = load_model(MODELS[name] / "checkpoint-15000", "triton", dev)
        scale = (cfg.n_embd // cfg.n_head) ** -0.5
        agg = {}
        for n_ex, ex in enumerate(examples(HOTPOT[a.L], a.L, a.n)):
            ids, is_sup, ctx_end = P.render_with_support(ex, tok)
            tc = TritonCapture(); tc.install()
            try:
                with torch.no_grad(): model(input_ids=ids.to(dev))
            finally: tc.uninstall()
            if len(tc.layers) != cfg.n_layer: raise RuntimeError(f"{name}: captured {len(tc.layers)} layers")
            C, E = rows_by_recurrence(tc.layers)
            for l in range(cfg.n_layer):
                agg.setdefault(l, []).append(P.stats_one(C[l], C[l] - E[l], torch.zeros_like(C[l]), is_sup, ctx_end, scale))
            print(f"{name} ex {n_ex} T={ids.shape[1]}", flush=True)
        res = {"model": name, "L": a.L, "n": a.n, "bins": P.BINS, "bias_layers": 0, "kernel": "triton",
               "layers": {str(l): P.reduce(v) for l, v in sorted(agg.items())}}
        f.write_text(json.dumps(res, indent=1)); print(f"[done] {f.name}", flush=True)
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

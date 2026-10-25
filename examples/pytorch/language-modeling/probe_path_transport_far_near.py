#!/usr/bin/env python3
"""PAT-310: what does PaTH transport carry from far keys vs near keys?

PaTH base logit (UT form, path_ut_base_raw):  E = q.k  -  correction
  C = lower_QK   : plain content match q_i . k_j
  R = correction : cumulative-Householder transport term (M_base @ strictLower(W K^T))
  E = C - R      : what the query actually sees through the path
  bias           : QWAB / prefix_box additive term (E_wav - E), 0 for PaTH
All in final logit units (x d^-0.5).

For answer-side query rows (last Q_ROWS positions: question + "Answer:"), per layer:
  (a) by distance bin d=i-j: rms(C), rms(R), rms(E), corr(C,E) [content fidelity
      through transport], base/full attention mass per bin.
  (b) support retrieval: support-doc keys vs distance-matched distractor keys,
      SNR_X = (mean_S X - mean_D X) / std_D X for X in {C, -R, E, E+bias},
      and attention mass on support (base softmax vs full softmax).
Front-placed HotpotQA-Long (hotpot_long_dev.jsonl), so support is far at L>=2048.
"""
import argparse, ast, json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/project/nlp-work5/hongyu-s/transformers/src")
sys.path.insert(0, "/project/nlp-work5/hongyu-s/flash-linear-attention")
import fla.layers.path_attn as pa
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

R = Path("/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs")
HOTPOT = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/hotpot_long_dev.jsonl"
MODELS = {
    "PaTH_s42": R / "PA_baseline_multi_seeds/token_even_mix_PA_s42",
    "PaTH_s43": R / "PA_baseline_multi_seeds/s43",
    "PaTH_s44": R / "PA_baseline_multi_seeds/s44",
    "QWAB_s42": R / "pat244_dual_temp/K1_L512_me16_rho256_ricker_rerun_s42",
    "QWAB_s43": R / "pat244_dual_temp/K1_L512_me16_rho256_seed43",
    "QWAB_s44": R / "pat244_dual_temp/K1_L512_me16_rho256_seed44",
    "box055_s42": R / "pat310_c2/K1_L512_me16_rho256_prefix_box_w055_s42",
    "box055_s43": R / "pat310_c2/K1_L512_me16_rho256_prefix_box_w055_s43",
    "box055_s44": R / "pat310_c2/K1_L512_me16_rho256_prefix_box_w055_s44",
    "boxsb086_s42": R / "pat310_c2/K1_L512_me16_rho256_prefix_box_sb086_s42",
    "boxsb086_s43": R / "pat310_c2/K1_L512_me16_rho256_prefix_box_sb086_s43",
    "boxsb086_s44": R / "pat310_c2/K1_L512_me16_rho256_prefix_box_sb086_s44",
}
BINS = [1, 4, 16, 64, 256, 512, 1024, 2048, 4096, 8192]
Q_ROWS = 24
MATCH_PAD = 256
PREFIX = 512     # 2*rho: where the prefix bias lives (absolute key index)
NEAR = 512       # training length
PROFILE = 512
PBUCK = 16


def render_with_support(ex, tok):
    parts, spans, pos = ["Context:\n"], [], len("Context:\n")
    sup = {t for t, _ in ex["supporting_facts"]}
    for t, s in ex["context"]:
        d = f"Title: {t}\n{' '.join(s).strip()}\n\n"
        if t in sup: spans.append((pos, pos + len(d)))
        parts.append(d); pos += len(d)
    text = "".join(parts) + f"\nQuestion: {ex['question']}\nAnswer:"
    enc = tok(text, return_offsets_mapping=True)
    ids = enc["input_ids"]
    is_sup = np.zeros(len(ids), bool)
    ctx_end = 0
    for n, (a, b) in enumerate(enc["offset_mapping"]):
        if a < pos: ctx_end = n + 1
        for s0, s1 in spans:
            if a < s1 and b > s0: is_sup[n] = True
    return torch.tensor([ids]), is_sup, ctx_end


class Capture:
    def __init__(self):
        self.reset()

    def reset(self):
        self.calls = 0; self.base = {}; self.bias = {}

    def install(self):
        self.o_base, self.o_bias = pa.path_ut_base_raw, pa.PaTHAttention._build_ctxscale_shift_logit_bias_v0
        cap = self

        def base_raw(*a, **kw):
            out = cap.o_base(*a, **kw)
            E, _, _, _, C, Rr = out
            lid = cap.calls; cap.calls += 1
            T = E.shape[-1]
            sl = slice(T - Q_ROWS, T)
            cap.base[lid] = (C[0, :, sl, :].float().clone(), Rr[0, :, sl, :].float().clone())
            return out

        def bias_fn(self_, **kw):
            out = cap.o_bias(self_, **kw)
            E_wav = out[0] if isinstance(out, tuple) else out
            E = kw["E_base_raw"]; T = E.shape[-1]; sl = slice(T - Q_ROWS, T)
            cap.bias[cap.calls - 1] = (E_wav - E)[0, :, sl, :].float().clone()
            return out

        pa.path_ut_base_raw = base_raw
        pa.PaTHAttention._build_ctxscale_shift_logit_bias_v0 = bias_fn

    def uninstall(self):
        pa.path_ut_base_raw = self.o_base
        pa.PaTHAttention._build_ctxscale_shift_logit_bias_v0 = self.o_bias


def masked_softmax(x, valid):
    x = x.masked_fill(~valid, float("-inf"))
    return torch.softmax(x, dim=-1)


def stats_one(C, Rr, Bv, is_sup, ctx_end, scale):
    """C,Rr,Bv: [H,Q,T] raw. Returns dict of per-bin and support metrics (mean over heads&rows)."""
    H, Q, T = C.shape
    dev = C.device
    C = C * scale; Rr = Rr * scale; Bv = Bv * scale
    E = C - Rr; F = E + Bv
    i = torch.arange(T - Q, T, device=dev).view(1, Q, 1)
    j = torch.arange(T, device=dev).view(1, 1, T)
    dist = (i - j).expand(H, Q, T)
    valid = dist >= 0
    pE, pF = masked_softmax(E, valid), masked_softmax(F, valid)
    out = {"bin": []}
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (dist >= lo) & (dist < hi)
        if m.sum() < H * Q * 2: out["bin"].append(None); continue
        mf = m.float(); n = mf.sum(-1).clamp_min(1)
        def rms(x): return float(((x ** 2 * mf).sum(-1) / n).sqrt().mean())
        def cen(x):
            mu = (x * mf).sum(-1, keepdim=True) / n.unsqueeze(-1)
            return (x - mu) * mf
        cc, ce = cen(C), cen(E)
        corr = (cc * ce).sum(-1) / ((cc ** 2).sum(-1).sqrt() * (ce ** 2).sum(-1).sqrt()).clamp_min(1e-9)
        rr = cen(Rr)
        out["bin"].append(dict(
            rmsC=rms(cen(C)), rmsR=rms(rr), rmsE=rms(ce), rmsB=rms(cen(Bv)),
            corrCE=float(corr.mean()),
            massE=float((pE * mf).sum(-1).mean()), massF=float((pF * mf).sum(-1).mean()),
        ))
    jj = j.expand(H, Q, T)
    regions = {"prefix": valid & (jj < PREFIX), "far_mid": valid & (jj >= PREFIX) & (dist >= NEAR),
               "near": valid & (jj >= PREFIX) & (dist < NEAR)}
    out["region"] = {}
    for rn, m in regions.items():
        if m.sum() < H * Q * 2: continue
        mf = m.float(); n = mf.sum(-1).clamp_min(1)
        def mean(x): return (x * mf).sum(-1) / n
        def cen(x): return (x - mean(x).unsqueeze(-1)) * mf
        def rms(x): return float(((x ** 2 * mf).sum(-1) / n).sqrt().mean())
        def corr(a, b):
            a, b = cen(a), cen(b)
            return float(((a * b).sum(-1) / ((a ** 2).sum(-1).sqrt() * (b ** 2).sum(-1).sqrt()).clamp_min(1e-9)).mean())
        out["region"][rn] = dict(
            rmsC=rms(cen(C)), rmsR=rms(cen(Rr)), rmsE=rms(cen(E)), rmsB=rms(cen(Bv)),
            meanE=float(mean(E).mean()), meanB=float(mean(Bv).mean()), meanR=float(mean(Rr).mean()),
            corrCE=corr(C, E), corrBR=corr(Bv, Rr), corrBE=corr(Bv, E), corrBC=corr(Bv, C),
            massE=float((pE * mf).sum(-1).mean()), massF=float((pF * mf).sum(-1).mean()),
        )
    # absolute-position profile of the first PROFILE keys (bucketed), mean over heads/rows
    nb_ = PROFILE // PBUCK
    def prof(x): return x[:, :, :PROFILE].mean((0, 1)).view(nb_, PBUCK).mean(-1).tolist()
    out["profile"] = dict(E=prof(E), R=prof(Rr), C=prof(C), B=prof(Bv),
                          pE=prof(pE * PBUCK), pF=prof(pF * PBUCK))
    sup = torch.tensor(is_sup, device=dev).view(1, 1, T).expand(H, Q, T) & valid
    if sup.any():
        sd = dist[sup]; dlo, dhi = int(sd.min()) - MATCH_PAD, int(sd.max()) + MATCH_PAD
        ctx = torch.zeros(T, dtype=torch.bool, device=dev); ctx[:ctx_end] = True
        dis = ctx.view(1, 1, T).expand(H, Q, T) & ~sup & valid & (dist >= dlo) & (dist <= dhi)
        s_f, d_f = sup.float(), dis.float()
        def snr(x):
            ms = (x * s_f).sum(-1) / s_f.sum(-1).clamp_min(1)
            md = (x * d_f).sum(-1) / d_f.sum(-1).clamp_min(1)
            vd = (((x - md.unsqueeze(-1)) ** 2) * d_f).sum(-1) / d_f.sum(-1).clamp_min(1)
            z = (ms - md) / vd.sqrt().clamp_min(1e-9)
            return float(z.mean()), float(z.max(0).values.mean())  # mean over heads, best head
        out["sup"] = dict(
            dist_mean=float(sd.float().mean()),
            snrC=snr(C), snrNegR=snr(-Rr), snrE=snr(E), snrF=snr(F),
            massE=float((pE * s_f).sum(-1).mean()), massF=float((pF * s_f).sum(-1).mean()),
            massE_best=float((pE * s_f).sum(-1).mean(1).max()), massF_best=float((pF * s_f).sum(-1).mean(1).max()),
            sup_frac=float(s_f[0, 0].sum() / valid[0, 0].sum()),
        )
    return out


def apply_supply_cfg(cfg, run_dir):
    """Mirror run_clm.py --cfg_path: add missing keys, force-override wavelet_* prefixes."""
    f = run_dir / "supply_model.cfg"
    if not f.exists(): return
    kv = {}
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        k, v = (x.strip() for x in line.split("=", 1))
        try: v = ast.literal_eval(v)
        except Exception: pass
        kv[k] = v
    have = set(cfg.to_dict())
    for k, v in kv.items():
        if k not in have or k.startswith(("wavelet_mode", "wavelet_ctxscale", "wavelet_logit_bias", "wavelet_router_sigmoid",
                                          "wavelet_ctx_feat", "rel_use_layer", "_attn_implementation")) or k == "rel_use_layer_list":
            setattr(cfg, k, v)


def run_model(name, ck, tok, L, n, dev, cap):
    cfg = AutoConfig.from_pretrained(ck); apply_supply_cfg(cfg, ck.parent)
    if not hasattr(cfg, "wavelet_ctxscale_k"):  # PaTH baseline: ckpt carries unused K=8 ctxscale params (db1, no router)
        cfg.wavelet_ctxscale_k = 8
        mx = getattr(cfg, "wavelet_ctxscale_scale_max_exp", 14.0)
        if not isinstance(mx, (list, tuple)): cfg.wavelet_ctxscale_scale_max_exp = [float(mx)] * 8
    cfg.attn_implementation = "path_attn"; cfg.path_attn_impl = "pytorch"
    cfg.wavelet_logit_bias_log_every = 10 ** 9; cfg.wavelet_ctxscale_eval_log_once = False
    model = AutoModelForCausalLM.from_pretrained(ck, config=cfg, torch_dtype=torch.float32).to(dev).eval()
    for m in model.modules():
        if isinstance(m, pa.PaTHAttention): m._capture_debug_tensors = False
    hd = cfg.n_embd // cfg.n_head; scale = hd ** -0.5
    exs = []
    with open(HOTPOT) as f:
        for line in f:
            ex = json.loads(line)
            if ex["meta"]["target_total_tokens"] == L: exs.append(ex)
            if len(exs) >= n: break
    agg = {}
    for ex in exs:
        ids, is_sup, ctx_end = render_with_support(ex, tok)
        cap.reset()
        with torch.no_grad(): model(input_ids=ids.to(dev))
        if cap.calls != cfg.n_layer: raise RuntimeError(f"{name}: path_ut_base_raw called {cap.calls}x, expected {cfg.n_layer}")
        for lid, (C, Rr) in cap.base.items():
            Bv = cap.bias.get(lid, torch.zeros_like(C))
            agg.setdefault(lid, []).append(stats_one(C, Rr, Bv, is_sup, ctx_end, scale))
    nb = len(cap.bias)
    del model; torch.cuda.empty_cache()
    return agg, nb


def reduce(lst):
    """average list of stats dicts (per sample) -> one dict."""
    out = {"bin": []}
    for b in range(len(BINS) - 1):
        vals = [s["bin"][b] for s in lst if s["bin"][b] is not None]
        out["bin"].append({k: float(np.mean([v[k] for v in vals])) for k in vals[0]} if vals else None)
    out["region"] = {}
    for rn in ("prefix", "far_mid", "near"):
        vals = [s["region"][rn] for s in lst if rn in s["region"]]
        if vals: out["region"][rn] = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
    out["profile"] = {k: np.mean([s["profile"][k] for s in lst], 0).tolist() for k in lst[0]["profile"]}
    sv = [s["sup"] for s in lst if "sup" in s]
    if sv:
        out["sup"] = {}
        for k in sv[0]:
            if isinstance(sv[0][k], tuple): out["sup"][k] = tuple(float(np.mean([v[k][t] for v in sv])) for t in range(2))
            else: out["sup"][k] = float(np.mean([v[k] for v in sv]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="PaTH_s42,QWAB_s42,box055_s42")
    ap.add_argument("--lengths", default="512,2048,4096")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--out", default=str(R / "pat310_c2/transport_probe"))
    a = ap.parse_args()
    dev = torch.device("cuda"); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("gpt2")
    cap = Capture(); cap.install()
    try:
        for name in a.models.split(","):
            ck = MODELS[name] / "checkpoint-15000"
            if not ck.exists(): print(f"[skip] {name}"); continue
            for L in map(int, a.lengths.split(",")):
                f = out / f"{name}_L{L}.json"
                if f.exists(): print(f"[have] {f.name}"); continue
                agg, nb = run_model(name, ck, tok, L, a.n, dev, cap)
                res = {"model": name, "L": L, "n": a.n, "bins": BINS, "bias_layers": nb,
                       "layers": {str(l): reduce(v) for l, v in sorted(agg.items())}}
                f.write_text(json.dumps(res, indent=1)); print(f"[done] {f.name} bias_layers={nb}", flush=True)
    finally:
        cap.uninstall()


if __name__ == "__main__":
    main()

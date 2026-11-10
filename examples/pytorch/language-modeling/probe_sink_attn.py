#!/usr/bin/env python3
"""PAT-310 RC4: position-agnostic attention-sink measures, PaTH vs QWAB, GPT-2 small (follow-up of probe_first_token_attn.py).

Sinks are not only the first token: Xiao et al. (ICLR 2024) keep the four initial tokens; Sun et al. (COLM 2024) find
massive activations / attention concentration also on delimiter tokens ('.', '\\n', ',') and weak-semantics words.
Per layer we record, over the last W=256 query positions (question + answer-generating region) and at the answer position
(last prompt token):
  * first4   : attention share on the first 4 tokens (Xiao et al.)
  * maxkey   : attention on the single most-attended key, averaged over queries (concentration, position-agnostic)
  * top-4 received keys per head among context keys outside the query window (j < T-W): r_j = mean_i p_ij over the
    window queries; record their total share, positions and token ids (to see whether they are delimiters / BOS).
Probabilities: the softmaxed P_wav used by the PaTH attention output (pytorch impl). Prompts: front-pinned HotpotQA-Long
eval prompt, first n examples per length (same as probe_first_token_attn.py / probe_gate_by_offset.py).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

sys.argv0 = list(sys.argv); sys.argv = ["x"]
sys.path.insert(0, "/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling")
sys.path.insert(0, "/project/nlp-work5/hongyu-s/transformers/src")
sys.path.insert(0, "/project/nlp-work5/hongyu-s/flash-linear-attention")
import run_clm as rc
import fla.layers.path_attn as pa
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from probe_path_transport_far_near import MODELS, apply_supply_cfg
sys.argv = sys.argv0

HOTPOT = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/hotpot_long_dev.jsonl"
W = 256


def build_prompt(ex, L, tok):
    pre = tok("Context:\n", add_special_tokens=True)["input_ids"]
    q_ids = tok(f"\nQuestion: {ex['question']}\n", add_special_tokens=False)["input_ids"]
    a_ids = tok("Answer:", add_special_tokens=False)["input_ids"]
    budget = L - len(pre) - len(q_ids) - len(a_ids) - 16
    txt, cids, st = rc.build_context_budgeted(ex, budget, prefer_same_title=True, min_tokens=budget - 64,
                                              support_offset_range=(0, 0))
    return torch.tensor([pre + list(cids) + q_ids + a_ids])


CAP = {}
CUR = {"ids": None}


def cap(self, probs, future, layer_idx):
    lid = int(self.layer_idx if layer_idx is None else layer_idx)
    assert probs.dim() == 4 and probs.shape[0] == 1 and probs.shape[-2] == probs.shape[-1], tuple(probs.shape)
    p = probs[0].detach().float()                          # [H, T, T]
    T = p.shape[-1]
    w = min(W, T // 2)
    q = p[:, T - w:, :]                                    # [H, w, T] late query window
    a = p[:, T - 1, :]                                     # [H, T] answer position
    r = q[:, :, :T - w].mean(1)                            # [H, T-w] received attention of context keys
    top = r.topk(4, dim=-1)
    pos = top.indices                                      # [H, 4]
    ids = CUR["ids"]
    CAP[lid] = dict(first4_late=q[:, :, :4].sum(-1).mean().item(),
                    first4_ans=a[:, :4].sum(-1).mean().item(),
                    maxkey_late=q.max(-1).values.mean().item(),
                    maxkey_ans=a.max(-1).values.mean().item(),
                    top4_share=top.values.sum(-1).tolist(),
                    top4_share_ans=torch.gather(a[:, :T - w], 1, pos).sum(-1).tolist(),
                    top4_pos=pos.tolist(),
                    top4_tok=[[int(ids[j]) for j in row] for row in pos.tolist()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="PaTH_s42,PaTH_s43,PaTH_s44,QWAB_s42,QWAB_s43,QWAB_s44")
    ap.add_argument("--lens", default="512,2048,4096")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="/cl/work5/hongyu-s/tmp_pat310/sink_attn")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    rc.GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained("gpt2"); tok = rc._get_tokenizer()
    rows = {}
    with open(HOTPOT) as f:
        for line in f:
            ex = json.loads(line); rows.setdefault(ex["meta"]["target_total_tokens"], []).append(ex)
    pa.PaTHAttention._maybe_capture_entmax_attention_stats = cap
    for name in a.models.split(","):
        ck = MODELS[name] / "checkpoint-15000"
        cfg = AutoConfig.from_pretrained(ck); apply_supply_cfg(cfg, ck.parent)
        if not hasattr(cfg, "wavelet_ctxscale_k"):  # PaTH baseline: ckpt carries unused K=8 ctxscale params
            cfg.wavelet_ctxscale_k = 8
            mx = getattr(cfg, "wavelet_ctxscale_scale_max_exp", 14.0)
            if not isinstance(mx, (list, tuple)): cfg.wavelet_ctxscale_scale_max_exp = [float(mx)] * 8
        cfg.attn_implementation = "path_attn"; cfg.path_attn_impl = "pytorch"
        cfg.wavelet_logit_bias_log_every = 10 ** 9; cfg.wavelet_ctxscale_eval_log_once = False
        model = AutoModelForCausalLM.from_pretrained(ck, config=cfg, torch_dtype=torch.float32).to(dev).eval()
        layers = [m for m in model.modules() if isinstance(m, pa.PaTHAttention)]
        for m in layers: m._capture_debug_tensors = False
        for L in map(int, a.lens.split(",")):
            f = out / f"{name}_L{L}.json"
            if f.exists(): print(f"[have] {f.name}"); continue
            recs = []
            for ex in rows[L][:a.n]:
                ids = build_prompt(ex, L, tok)
                CUR["ids"] = ids[0].tolist(); CAP.clear()
                with torch.no_grad(): model(input_ids=ids.to(dev))
                assert sorted(CAP) == list(range(len(layers))), (name, L, sorted(CAP))
                recs.append(dict(T=int(ids.shape[1]), layers=[CAP[l] for l in range(len(layers))]))
            f.write_text(json.dumps({"model": name, "L": L, "W": W, "recs": recs}))
            mk = np.mean([[l["maxkey_late"] for l in r["layers"]] for r in recs])
            t4 = np.mean([[np.mean(l["top4_share"]) for l in r["layers"]] for r in recs])
            f4 = np.mean([[l["first4_late"] for l in r["layers"]] for r in recs])
            print(f"[done] {f.name} n={len(recs)} first4={f4:.4f} maxkey={mk:.4f} top4_received_share={t4:.4f}", flush=True)
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

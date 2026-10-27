#!/usr/bin/env python3
"""PAT-310 W5 sanity: how much does the PaTH-FoX forget gate decay distant keys?

FoX logit(i,j) += sum_{t=j+1..i} log f_t, with log f_t = logsigmoid(g_proj(h_t)) per head.
On HotpotQA-Long inputs we record log f per layer/head/position (answer-side context) and report,
per layer, the head-wise distribution of the cumulative decay over the last D tokens before the
answer, for D in {64, 512, 2048, 3000}. Heads with decay < -10 nats cannot attend that far at all.
Also compares the pretrained PaTH-FoX (before finetune) with the finetuned checkpoints.
"""
import json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/project/nlp-work5/hongyu-s/transformers/src")
sys.path.insert(0, "/project/nlp-work5/hongyu-s/flash-linear-attention")
import fla.layers.path_attn as pa
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, str(Path(__file__).parent))
from probe_path_transport_far_near import apply_supply_cfg

R = Path("/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs")
HOTPOT = "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/data/hotpot_long_dev.jsonl"
MODELS = {
    "FoX_pretrain": R / "1r_FoX_baseline_from_s",
    "FoX_ft_s42": R / "pat310_fox/small_pathfox_s42/checkpoint-15000",
    "FoX_ft_s43": R / "pat310_fox/small_pathfox_s43/checkpoint-15000",
    "FoX_ft_s44": R / "pat310_fox/small_pathfox_s44/checkpoint-15000",
}
DS = [64, 512, 2048, 3000]
N = 8


def render(ex):
    c = "".join(f"Title: {t}\n{' '.join(s).strip()}\n\n" for t, s in ex["context"])
    return f"Context:\n{c}\nQuestion: {ex['question']}\nAnswer:"


def main():
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained("gpt2")
    exs = [json.loads(l) for l in open(HOTPOT)]
    exs = [e for e in exs if e["meta"]["target_total_tokens"] == 4096][:N]
    out = {}
    for name, ck in MODELS.items():
        cfg = AutoConfig.from_pretrained(MODELS["FoX_ft_s42"])  # full run_clm-resolved config; same arch for all four
        apply_supply_cfg(cfg, R / "pat310_fox")  # supply_model.cfg-equivalent: wavelet off, K=1
        for k, v in dict(scale_range=[0, 16], sample_num=16, rel_selection="all", wavelet_mode="off",
                         wavelet_ctxscale_k=1, wavelet_ctxscale_scale_max_exp=16).items():
            if not hasattr(cfg, k) or k.startswith("wavelet"): setattr(cfg, k, v)
        cfg.attn_implementation = "path_attn"; cfg.path_attn_impl = "triton"; cfg.use_forget_gate = True
        model = AutoModelForCausalLM.from_pretrained(ck, config=cfg, torch_dtype=torch.float32).to(dev).eval()
        mods = [m for m in model.modules() if isinstance(getattr(m, "g_proj", None), torch.nn.Linear) and getattr(m, "use_forget_gate", False)]
        print(f"{name}: {len(mods)} forget-gate modules ({type(mods[0]).__name__ if mods else '-'})", flush=True)
        assert mods, "no forget-gate modules found"
        store = {}
        hooks = []
        for li, m in enumerate(mods):
            def mk(li):
                def h(mod, inp, outp):
                    store[li] = torch.nn.functional.logsigmoid(outp.float()).detach()[0].cpu()  # [T, H]
                return h
            hooks.append(m.g_proj.register_forward_hook(mk(li)))
        acc = {li: {d: [] for d in DS} for li in range(len(mods))}
        for ex in exs:
            ids = tok(render(ex), return_tensors="pt")["input_ids"][:, :4200].to(dev)
            store.clear()
            with torch.no_grad(): model(input_ids=ids)
            for li, lf in store.items():
                T = lf.shape[0]
                for d in DS:
                    if d < T: acc[li][d].append(lf[T - d:].sum(0).numpy())  # [H] cumulative log-decay
        for hk in hooks: hk.remove()
        res = {}
        for li in acc:
            res[li] = {}
            for d in DS:
                a = np.mean(acc[li][d], 0)  # mean over samples, per head
                res[li][d] = dict(median=float(np.median(a)), max=float(a.max()),
                                  n_heads_gt_m5=int((a > -5).sum()), n_heads=int(a.size))
        out[name] = res
        print(f"== {name}  (per layer: #heads with cum log-decay > -5 nats, out of H; median over heads)")
        for li in res:
            print(f"L{li:2d} " + "  ".join(f"d={d}: {res[li][d]['n_heads_gt_m5']}/{res[li][d]['n_heads']} med={res[li][d]['median']:.1f} max={res[li][d]['max']:.1f}" for d in DS), flush=True)
        del model; torch.cuda.empty_cache()
    Path(R / "pat310_fox/forget_decay_probe.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

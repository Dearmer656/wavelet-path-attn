#!/usr/bin/env python3
"""finetune_nope_launcher.py — PAT-204.

Continue-finetune the rotary GPT-2 with RoPE switched OFF on the 8 PAT-202
vertical heads, by wrapping `AutoModelForCausalLM.from_pretrained` at import time
to apply the per-head NoPE monkey-patch, then handing control to run_clm.py
unchanged (so we reuse its exact `mix` data pipeline). Zero edits to run_clm.py
or modeling_gpt2.py.

Controlled by env NOPE_PRESET in {none, vertical}:
  none      all-RoPE continue-finetune (control)
  vertical  RoPE off on the 8 vertical heads (treatment)

Usage (run from .../language-modeling, so `run_clm` is importable):
  NOPE_PRESET=vertical python hotpot_long/finetune_nope_launcher.py <run_clm args...>
"""

import os
import re
import sys

# make run_clm importable (this file lives in hotpot_long/, run_clm.py one level up)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoModelForCausalLM  # noqa: E402
from eval_nope_vertical import apply_nope, apply_lowfreq_nope, VERTICAL_HEADS  # noqa: E402

_PRESET = os.environ.get("NOPE_PRESET", "none")
_orig_from_pretrained = AutoModelForCausalLM.from_pretrained


def _patched_from_pretrained(*args, **kwargs):
    ret = _orig_from_pretrained(*args, **kwargs)
    # run_clm calls with output_loading_info=True -> (model, info) tuple
    model = ret[0] if isinstance(ret, tuple) else ret
    if _PRESET == "vertical":
        nl = int(getattr(model.config, "n_layer", 12))
        nh = int(getattr(model.config, "n_head", 12))
        # priority: NOPE_HEADS_FILE (explicit "layer head" lines) > NOPE_LAYERS
        # (all heads of given layers, dash-separated to survive --export) > the 8
        # PAT-202 vertical heads.
        heads_file = os.environ.get("NOPE_HEADS_FILE", "")
        nope_layers = [x for x in re.split(r"[^0-9]+", os.environ.get("NOPE_LAYERS", "")) if x]
        if heads_file:
            head_set = set()
            for line in open(heads_file):
                p = line.split()
                if len(p) >= 2:
                    head_set.add((int(p[0]), int(p[1])))
            desc = f"explicit head set from {os.path.basename(heads_file)}"
        elif nope_layers:
            layers = [int(x) for x in nope_layers]
            head_set = {(l, h) for l in layers for h in range(nh)}
            desc = f"all heads of layers {layers}"
        else:
            head_set = set(VERTICAL_HEADS)
            desc = "the 8 PAT-202 vertical heads"
        n = apply_nope(model, head_set, nl, nh)
        print(f"[PAT-204] NoPE applied to {len(head_set)} heads ({desc}) "
              f"across {n} layers (rotate-then-restore on q/k)", flush=True)
    elif _PRESET == "lowfreq":
        # PAT-208 follow-up: switch RoPE off ONLY on the low-freq (long-period
        # P_m>cutoff) dims — the period-underexposed dims that produce the OOD
        # spike — keeping the high-freq (well-trained) dims on standard RoPE.
        nl = int(getattr(model.config, "n_layer", 12))
        nh = int(getattr(model.config, "n_head", 12))
        nope_layers = [int(x) for x in re.split(r"[^0-9]+", os.environ.get("NOPE_LAYERS", "")) if x]
        if not nope_layers:
            nope_layers = list(range(nl))
        cutoff = float(os.environ.get("NOPE_PERIOD_CUTOFF", "512"))
        n, n_off, periods = apply_lowfreq_nope(model, nope_layers, cutoff, nl, nh)
        print(f"[PAT-208] low-freq NoPE (P>{cutoff}) on {n_off}/{len(periods)} pairs "
              f"({2*n_off}/{len(periods)*2} dims) across layers {nope_layers} ({n} patched)",
              flush=True)
    elif _PRESET == "none":
        print("[PAT-204] NOPE_PRESET=none — all-RoPE control finetune", flush=True)
    else:
        raise ValueError(f"unknown NOPE_PRESET={_PRESET!r}")
    return ret


AutoModelForCausalLM.from_pretrained = staticmethod(_patched_from_pretrained)

import run_clm  # noqa: E402

if __name__ == "__main__":
    run_clm.main()

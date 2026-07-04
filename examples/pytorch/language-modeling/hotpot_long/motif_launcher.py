#!/usr/bin/env python3
"""motif_launcher.py — PAT-217. Apply motif substitution at model-load, then run_clm.

Wraps AutoModelForCausalLM.from_pretrained to call apply_motif_substitution on the
reconstruction-error-selected broken heads, then hands control to run_clm.py unchanged
(reuse the canonical teacher-forced HotpotQA F1 eval). Env-gated:
  MOTIF_NPZ       distilled_bias_L512.npz
  MOTIF_RECON_CSV recon_error_L2048.csv
  MOTIF_TOPK      number of broken heads (default 16)
  MOTIF_LAM       motif-bias scale (default 1.0)
  MOTIF_MODE      real | shuffled | wrong_head | random | none
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoModelForCausalLM  # noqa: E402
from motif_substitution import apply_motif_substitution, select_broken_heads  # noqa: E402

_MODE = os.environ.get("MOTIF_MODE", "real")      # off | nope_only | real | shuffled_* | wrong_head | random_matched
_NPZ = os.environ.get("MOTIF_NPZ", "")
_CSV = os.environ.get("MOTIF_RECON_CSV", "")
_TOPK = int(os.environ.get("MOTIF_TOPK", "16"))
_LAM = float(os.environ.get("MOTIF_LAM", "1.0"))
_TAIL = os.environ.get("MOTIF_TAIL", "hold_last")
_MIN_VE = float(os.environ.get("MOTIF_MIN_TRAIN_VE", "0.3"))
_orig = AutoModelForCausalLM.from_pretrained


def _patched(*args, **kwargs):
    ret = _orig(*args, **kwargs)
    model = ret[0] if isinstance(ret, tuple) else ret
    if _MODE == "off":
        print("[PAT-217] MOTIF_MODE=off — unmodified all-RoPE baseline", flush=True)
        return ret
    nl = int(getattr(model.config, "n_layer", 12))
    nh = int(getattr(model.config, "n_head", 12))
    import re as _re
    _layers_env = os.environ.get("MOTIF_LAYERS", "")
    if _layers_env:                              # explicit layers: ALL heads of these layers (PAT-204 style)
        layers = [int(x) for x in _re.split(r"[^0-9]+", _layers_env) if x]
        broken = [(l, h) for l in layers for h in range(nh)]
        print(f"[PAT-217] broken heads = ALL heads of layers {layers} ({len(broken)} heads)", flush=True)
    else:
        broken = select_broken_heads(_CSV, _TOPK, npz_path=_NPZ, min_train_ve=_MIN_VE, nh=nh)
        print(f"[PAT-217] selected broken heads (train_ve>={_MIN_VE}, top{_TOPK}): {broken}", flush=True)
    apply_motif_substitution(model, _NPZ, broken, lam=_LAM, mode=_MODE, nl=nl, nh=nh, tail=_TAIL)
    return ret


AutoModelForCausalLM.from_pretrained = staticmethod(_patched)

import run_clm  # noqa: E402

if __name__ == "__main__":
    run_clm.main()

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

_MODE = os.environ.get("MOTIF_MODE", "real")
_NPZ = os.environ.get("MOTIF_NPZ", "")
_CSV = os.environ.get("MOTIF_RECON_CSV", "")
_TOPK = int(os.environ.get("MOTIF_TOPK", "16"))
_LAM = float(os.environ.get("MOTIF_LAM", "1.0"))
_orig = AutoModelForCausalLM.from_pretrained


def _patched(*args, **kwargs):
    ret = _orig(*args, **kwargs)
    model = ret[0] if isinstance(ret, tuple) else ret
    if _MODE == "off":
        print("[PAT-217] MOTIF_MODE=off — unmodified baseline", flush=True)
        return ret
    nl = int(getattr(model.config, "n_layer", 12))
    nh = int(getattr(model.config, "n_head", 12))
    broken = select_broken_heads(_CSV, _TOPK)
    apply_motif_substitution(model, _NPZ, broken, lam=_LAM, mode=_MODE, nl=nl, nh=nh)
    return ret


AutoModelForCausalLM.from_pretrained = staticmethod(_patched)

import run_clm  # noqa: E402

if __name__ == "__main__":
    run_clm.main()

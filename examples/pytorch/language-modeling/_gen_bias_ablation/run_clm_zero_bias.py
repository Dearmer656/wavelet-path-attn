#!/usr/bin/env python3
"""Inference-only wavelet-bias ablation wrapper around run_clm.py.

Monkeypatches PaTHAttention._build_ctxscale_shift_logit_bias_v0 so the
returned attention logits are exactly E_base_raw (pure PaTH, no wavelet
bias added), while still calling the real implementation first so any
internal state / logging side effects behave normally. Everything else
(all trained weights, including the router/shift/gate params) is left
untouched -- this isolates whether the wavelet bias term is actively
used at inference time, on an already-trained QWAB checkpoint, with no
retraining involved.

Usage: identical CLI to run_clm.py (same argv), just invoke this file instead.
"""
import sys

sys.path.insert(0, "/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling")
sys.path.insert(0, "/project/nlp-work5/hongyu-s/transformers/src")
sys.path.insert(0, "/project/nlp-work5/hongyu-s/flash-linear-attention")

from fla.layers.path_attn import PaTHAttention  # noqa: E402

_orig_build_ctxscale = PaTHAttention._build_ctxscale_shift_logit_bias_v0


def _zero_bias_build_ctxscale(self, *, E_base_raw, **kwargs):
    _, payload = _orig_build_ctxscale(self, E_base_raw=E_base_raw, **kwargs)
    return E_base_raw, payload


PaTHAttention._build_ctxscale_shift_logit_bias_v0 = _zero_bias_build_ctxscale
print("[bias_ablation] PaTHAttention._build_ctxscale_shift_logit_bias_v0 patched: "
      "wavelet bias forced to zero, base PaTH logits passed through unchanged.", flush=True)

import run_clm  # noqa: E402

if __name__ == "__main__":
    run_clm.main()

#!/usr/bin/env python3
"""PAT-310: inference-time QWAB gate for the three wavelet-center placements (first position = QWAB, i/2, query).

Thin wrapper around probe_gate_by_offset.py: registers the center05 (i/2) and center10 (query-centered)
runs of pat310_center under new model names and runs the same probe (front-pinned eval prompt, first n
examples per length, gate at the answer position / question / context tokens). The QWAB results already
in tmp_pat310/gate_probe (QWAB_s4{2,3,4}_L{512,4096}_off0_0.json) are reused, not recomputed.
"""
import sys
from pathlib import Path

import probe_gate_by_offset as pgo

R = Path("/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/pat310_center")
for s in (42, 43, 44):
    pgo.MODELS[f"C05_s{s}"] = R / f"K1_L512_me16_rho256_ricker_center05_s{s}"
    pgo.MODELS[f"C10_s{s}"] = R / f"K1_L512_me16_rho256_ricker_center10_s{s}"

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--models", "C05_s42,C05_s43,C05_s44,C10_s42,C10_s43,C10_s44",
                "--conds", "512:0:0,4096:0:0", "--n", "100",
                "--out", "/cl/work5/hongyu-s/tmp_pat310/gate_probe"]
    pgo.main()

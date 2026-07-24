# K=1 single-scale results — MASTER TABLE

**Maintain this file.** New K=1 eval → fill the cell. Metric = HotpotQA-Long **uniform**
dev **F1** (autoregressive generation, `--path_attn_impl pytorch`), checkpoint-15000,
**seed 42** unless noted. `–` = not run yet.

**Naming:** `meX` = `wavelet_ctxscale_scale_max_exp = X`; single scale `ρ = 2^(X/2)`.
me0→ρ1, me2→ρ2, me4→ρ4, me6→ρ8, me8→ρ16, me10→ρ32, me12→ρ64, me14→ρ128,
me16→ρ256, me20→ρ1024, me22→ρ2048, me24→ρ4096, me26→ρ8192, me28→ρ16384.

**Variants:** A = norm-ON, center-OFF (standard). C = norm-ON, **center-ON** (PAT-234).
NOnorm = **norm-OFF**, center-OFF (raw wavelet).

---

## A — norm-ON, center-OFF  (`pat225_k1sc_S1_*`)

| ρ (meX) | L512 | L2048 | L4096 |
|---|---|---|---|
| 1 (me0) | 0.7652 | 0.7339 | 0.6799 |
| 2 (me2) | – | – | – |
| 4 (me4) | 0.7633 | 0.7207 | 0.6471 |
| 8 (me6) | – | – | – |
| 16 (me8) | 0.7638 | 0.7224 | 0.6524 |
| 32 (me10) | – | – | – |
| **64 (me12)** | **0.7659** | **0.7319** | **0.6699** |
| 128 (me14) | 0.7648 | 0.7302 | 0.6715 |
| 256 (me16) | 0.7588 | 0.7079 | 0.6433 |
| 1024 (me20) | 0.7646 | 0.7261 | 0.6616 |
| 2048 (me22) | – | – | – |
| 4096 (me24) | 0.7612 | 0.7259 | 0.6648 |
| 8192 (me26) | – | – | – |
| 16384 (me28) | 0.7643 | 0.7229 | 0.6594 |

*K=8-grid scales (ρ{1,4,16,64,256,1024,4096,16384}) all DONE + ρ128 center.
Intermediate scales ρ{2,8,32,2048,8192} (me2/6/10/22/26) training 2026-07-24, evals
auto-wired. ρ512 (me18) not requested. **All done scales beat PA-only@L4096 (0.6309),
range 0.643–0.680** — extrapolation gain is scale-robust; ρ1/ρ64/ρ128 lead.*

## C — norm-ON, center-ON  (`pat234_K1_*_C_s42`)

| ρ (meX) | L512 | L2048 | L4096 |
|---|---|---|---|
| 1 (me0) | – | – | – |
| 16 (me8) | – | – | – |
| 128 (me14) | 0.7569 | 0.7064 | – |
| 1024 (me20) | 0.7583 | 0.7201 | 0.6498 |
| 16384 (me28) | 0.7581 | 0.7062 | **0.6272** |

## NOnorm — norm-OFF, center-OFF  (`pat234_K1_*_NOnorm_s42`)

| ρ (meX) | L512 | L2048 | L4096 |
|---|---|---|---|
| 1 (me0) | – | – | 0.6628 |
| 16 (me8) | – | – | – |
| 128 (me14) | – | – | – |
| **256 (me16)** | – | – | 0.6511 |
| 1024 (me20) | – | – | 🔄 running |
| 16384 (me28) | – | – | – |

## REF — full models, 3-seed means (PAT-61)

| model | L512 | L1024 | L2048 | L3072 | L4096 |
|---|---|---|---|---|---|
| **PA-only** | 0.7625 | 0.7549 | 0.7114 | 0.6715 | **0.6309** |
| **QWAB (K=8)** | 0.7636 | 0.7575 | 0.7221 | 0.6868 | **0.6524** |
| Δ (QWAB−PA) | +0.11pp | +0.26 | +1.07✅ | +1.53✅ | **+2.15✅** |

---

## Caveats (read before interpreting)

0. **⚠️ FINE SCALES ARE GUTTED BY THE p99 CLAMP (probe_clamp_utilization.py, 2026-07-24).**
   per-scale RMS amplifies the sparse fine-scale spike (ρ1: 20.9×), then the p99 clamp
   threshold sits in the near-zero floor (99% of keys ≈0) and clips it away. Measured
   energy retained after p99: **ρ1 E_kept=0.000 (peak 334× over thr), ρ16=0.90,
   ρ256≈1.00, ρ16384≈1.00.** ⇒ the ρ1/ρ4 rows test ≈ "near-zero wavelet ≈ PA-only",
   NOT a fine-scale wavelet. So "ρ1 best (0.68)" is consistent with wavelet-inert, NOT
   with fine scales being good. Clamp2 (±g_bias_max=4.0) is essentially inert now
   (sat%≈0). Fixing this (clamp after sum / scale-aware thr / analytic norm) is required
   before any multi-scale free-selection design can let fine scales contribute.
1. **K=1 rows are single-seed (s42).** Per-scale/length differences ≲ 0.02 are within
   seed noise. Only effects > ~0.02 are trustworthy (e.g. C ρ16384 = −0.032 vs A).
2. **Do NOT compare K=1 F1 directly to PA-only.** K=1 ablations use `--pe_method no_pe`;
   PA_baseline uses vanilla PE → recipe confound. The clean "wavelet helps" number is
   the REF block (PA-only vs QWAB, **same recipe, 3-seed**): **+2.15pp @ L4096**,
   monotone-growing (length-targeted).
3. Reference amplification (analytic, 1/rawRMS @ L512, not F1):
   ρ1=20.9× · ρ16=6.8× · ρ128=2.4× · ρ256=1.8× · ρ1024=1.1× · ρ16384=1.0×.
   (norm amplifies fine/mid scales; ρ≥1024 ≈ untouched.)

## Robust conclusions so far

- **L512 flat** across scales (~0.764) — scale location irrelevant at training length.
- **Centering hurts at L4096**, worst for coarsest (ρ16384 −0.032) — only super-noise K=1 effect.
- **Wavelet helps extrapolation** (REF: +2.15pp @L4096, length-targeted) — the real result.

## Pending / to fill
- NOnorm: me20 (running), me8/me14/me28 (not run).
- C: me0/me8 L*, me14 L4096.
- Multi-seed (s43/s44) for any K=1 cell (all single-seed now).

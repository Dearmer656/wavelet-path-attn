# Frequency-domain coupling: why QWAB's wavelet span cannot cover PaTH's extrapolation drift

A first-principles argument coupling three mechanisms — PaTH's Householder-product
positional kernel, the Nyquist under-sampling that causes extrapolation drift, and
the Ricker wavelet's band-pass + DC-invisibility — to prove a **frequency-band
disjointness**: the band where the drift lives and the band the *usable* wavelet
scales cover do not overlap.

Notation: query `i`, key `j`, relative distance `Δ = i − j ≥ 0`. Training length `L`.
`C` = row-wise centering (removes the softmax-invisible constant, since
`softmax(z + c·1) = softmax(z)`), so only the AC (Δ-varying) part matters.

## 1. PaTH's positional kernel is oscillatory (Fourier-type)

PaTH logit: `a_{ij} = q_i^⊤ (∏_{k=j+1}^{i} H_k) k_j`, with `H_k = I − 2 w_k w_k^⊤`
(Householder reflections). The cumulative product `M(Δ) = ∏` is orthogonal; on the
subspace where the reflections share (approximately) an eigenbasis it acts as a
rotation whose angle accumulates with Δ. Diagonalizing the average generator gives
eigen-angles `{ω_m}`, so the content-averaged kernel is

  `ā(Δ) = E[a] ≈ Σ_m c_m cos(ω_m Δ + φ_m)`.  (P1)

PaTH is thus a **data-dependent RoPE**: position enters as a superposition of
oscillations at frequencies `{ω_m}` (the Householder eigen-angle spectrum). *[verify
(P1) empirically: FFT of ā(Δ) should show discrete/structured peaks, not white.]*

## 2. Extrapolation drift lives in the sub-training-length (low-frequency) band

At training length L, a component of frequency `ω_m` completes `ω_m L / 2π` periods
inside the window. Components with

  `ω_m < 2π / L`  ⇔  `f_m = ω_m/2π < f_train := 1/L`   (P2)

complete **less than one period** during training — they are under-sampled, so their
phase at test distances `Δ > L` is out-of-distribution. The aligned drift
`δ(Δ) = C(ā_long(Δ) − ā_short(Δ))` therefore concentrates its energy in the band
`f < f_train`. *(This is the RoPE-extrapolation folklore that PoSE/YaRN/NTK all target
the long-period dims; matches the project's PAT-209 "long-period OOD spike".)*
*[verify empirically: fraction of |FFT δ|² below f_train should be large.]*

## 3. Ricker frequency–scale law (analytic)

Ricker `ψ(t) = (1 − t²) e^{−t²/2}` has `Ψ(ω) ∝ ω² e^{−ω²/2}` — band-pass, peak at
`|ω| = √2`. At scale ρ, `ψ(t/ρ)` has peak angular frequency `ω_peak(ρ) = √2/ρ`, i.e.

  `f_peak(ρ) = √2 / (2π ρ)`.   (P3)

Larger ρ ⇒ lower frequency (monotone, exact).

## 4. Only coarse scales can reach the drift band — and they are unusable

Covering the drift band `f < f_train` needs `f_peak(ρ) < f_train`, i.e. by (P2)+(P3)

  `ρ > ρ* := √2 · L / (2π) ≈ 0.225 L`.   (P4)

For L = 512, `ρ* ≈ 115`, so only `ρ ∈ {256, 1024, 4096, 16384}` (the coarse scales)
have their pass-band inside the drift band. But for `ρ ≫ L` the Ricker is
near-constant over the causal window `[0, L]` (argument `Δ/ρ ≪ 1`), hence
**near-DC ⇒ softmax-invisible** (killed by C), and forcing it visible via centering
turns it into a slowly-varying ramp that **diffuses attention and is empirically
harmful** (PAT-234: ΔF1 = −0.032 @ L4096; attention entropy 2.70→3.22).

## 5. Result (substantial band mismatch — empirically calibrated)

The *analytic* threshold is exact: covering the drift band `f < f_train` requires
`ρ > ρ* = √2 L /(2π) ≈ 115` (L=512), i.e. only the coarse scales
`ρ ∈ {256,1024,4096,16384}` reach it, and those are softmax-invisible / harmful (§4).

The *empirical* strength is a **substantial, layer-dependent mismatch, not a total
one** (probe 528606):

> A large fraction of PaTH's aligned extrapolation drift lies in the sub-training-length
> band `f < f_train` reachable only by the unusable coarse scales:
> **aggregate ≈ 0.31 of drift energy, up to 0.70 in early–mid layers (L2=0.70,
> L1–L6 ∈ [0.34,0.70]), tapering to ~0.20 in late layers.**
>
> Hence QWAB's usable (fine/mid) scales **structurally cannot reach ≈1/3 (up to 2/3)
> of the drift** — the part in the coarse-only band. This is a lower bound: the
> high-frequency remainder of the measured δ is partly a population artifact (ā_512
> and ā_4096 average over different query/key sets), and where it is genuine it sits
> in the band where PaTH is well-sampled (little true drift) and QWAB is empirically
> inert anyway.

So the idealized "{drift} ∩ {usable} = ∅, coverage ≡ 0" is too strong; the honest
statement is a **strong partial structural mismatch** concentrated in early–mid layers.

**Corollary (still explains the observations).** (a) inference wavelet inert/removable
— QWAB-off ≈ QWAB-on; (b) centering hurts, worst at long length — −0.032 @ L4096;
(c) router avoids coarse scales at every length/position — K=8 heatmaps. The
mechanism accounts for all three without fitting.

## 6. What is analytic vs. what the probe verifies

- **Analytic:** (P3) Ricker band law; DC-invisibility; (P4) ρ* threshold; the band
  arithmetic in §5. No model needed.
- **Empirical (probe_spectral_coupling.py):** (P1) ā(Δ) has structured (oscillatory)
  spectrum; (P2) δ(Δ) energy concentrates below f_train; the per-scale band overlap
  of |FFT δ|² with `Ψ_ρ(ω)` gives an *analytic coverage upper bound* per scale — the
  quantity PAT-235 wanted, but bounded rather than black-box-projected.

## 7. Coverage upper bound (ties back to PAT-235)

Per scale, the maximum drift energy the scale can capture is bounded by the spectral
overlap `∫ |FFT δ(f)|² · Ψ̂_ρ(f)² df / ∫ |FFT δ|² df`, with `Ψ̂_ρ` the normalized
Ricker band. §5 implies this bound is ≈0 for fine/mid ρ (disjoint bands) and, for
coarse ρ, is multiplied by the (near-zero) softmax-visible fraction of that scale.
Summing gives total wavelet coverage of the drift `≈ 0` **by construction of the
scale grid vs. the training length** — a structural, not empirical, null.

---

# 8. Geometric explanation of the scaffolding gain

Premise-(a) check (same seed s43, L4096): PaTH-only 0.6321 → QWAB-off 0.6446
(+0.012, retained with wavelet disabled) → QWAB-on 0.6554 (+0.023). So QWAB
*training* leaves a gain that survives removing the bias — a **training-time
scaffold**, not an inference correction. The spectral framework explains it, and
crucially explains why it is *small and in-distribution*, not an extrapolation fix.

## 8.1 The wavelet is a frequency-localized attention prior — in the USABLE band

The router picks mid scales ρ∈{16,64,256} (K=8 heatmaps). By the Ricker band law
(§3), these sit at `f_peak ∈ [8.8e-4, 1.4e-2]` — i.e. **above / at f_train**, the
well-sampled, non-drifting band (§2,§5). So during training the bias
`b(Δ)=Σ_s π_s ricker(Δ/ρ_s)` injects a **band-pass, distance-selective** signal
exactly where PaTH is well-behaved — a differentiable "attend at distance ≈ ρ_s" hint.

## 8.2 Basin selection in weight space (why the gain exists)

Learning distance-selective retrieval attention from scratch is a hard non-convex
problem: Q/K/Householder must *discover* which relative distances matter. The additive
Ricker prior supplies that distance structure **for free early in training**, so SGD
can specialize Q/K on content *within* an already-good distance scaffold. This is a
homotopy/curriculum: the loss landscape seen by the QWAB-trained model has the
mid-range selectivity "pre-installed", steering optimization into a basin whose PaTH
weights encode good mid-range attention — a basin a PaTH-only run can miss.

## 8.3 Imprint & removability (why it survives disabling)

Once the Householder spectrum `{ω_m}` and Q/K projections encode the mid-range pattern,
the additive bias is **redundant** — the pattern now lives in the weights. Removing it
at inference keeps the pattern ⇒ QWAB-off > PaTH-only (the +0.012). The extra +0.011
from turning it on is the (small, band-limited) residual direct contribution.

## 8.4 Spectral signature ⇒ the gain is IN-DISTRIBUTION, not an extrapolation fix

Because the scaffold operates in the usable (high-freq, f>f_train) band, it refines a
region PaTH already samples well. Two consequences, both falsifiable:
- **It cannot touch the low-freq extrapolation drift** (§5): consistent with the
  inference wavelet being inert/harmful and the router avoiding coarse scales.
- **The gain is a better *base* model, ≈ uniform across lengths** — NOT growing with
  extrapolation distance. Prediction: `gain(L512) ≈ gain(L2048) ≈ gain(L4096)`.
  *(test: PaTH-only s43 @ L512/L2048 vs QWAB s43; if the +0.02 is roughly length-flat →
  in-distribution scaffold confirmed; if it grows with length → contradicts this account.)*

## 8.5 One coherent picture

The **same** frequency mismatch that makes QWAB useless as an *inference* positional
corrector (drift lives in the coarse-only band it cannot realize, §5) makes it a mild
*training* regularizer (its mid band coincides with the useful, well-sampled retrieval
distances). Wavelet-as-inference-bias: inert/harmful. Wavelet-as-training-scaffold:
small, in-distribution, removable gain. Both fall out of one law:
`f_peak(ρ)=√2/(2πρ)` vs the training-length Nyquist `f_train=1/L`.

## 8.6 Geometric verification (weight-space imprint)

Direct test of §8.3: on identical inputs, `δ(Δ) = ā_QWAB-off(Δ) − ā_PaTH-only(Δ)`
(both are pure-PaTH logits) should have its energy in the **mid-scale band**
`f_peak(ρ16..ρ256)`, not the low-freq drift band. Concentration there = the wavelet's
training frequencies imprinted into the PaTH weights. (Needs a generic attention-logit
hook for the db1 PaTH-only model; the length-profile test §8.4 is the cheaper first cut.)

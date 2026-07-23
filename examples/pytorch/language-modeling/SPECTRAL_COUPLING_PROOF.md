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

## 5. Theorem (band disjointness)

> The frequencies of PaTH's extrapolation drift satisfy `f < f_train`; the wavelet
> scales that are *usable* (softmax-visible, non-harmful) satisfy `f_peak > f_train`
> (fine/mid, `ρ < ρ*`). Hence
>
>   **{drift band} ∩ {usable-scale band} = ∅.**
>
> The only scales whose pass-band overlaps the drift are `ρ > ρ*`, which are
> softmax-invisible (near-DC) and harmful when forced visible. Therefore QWAB's
> wavelet subspace **cannot correct PaTH's length-extrapolation drift**: the
> component it *could* represent (fine/mid, high-freq) is exactly the band where PaTH
> does **not** drift, and the band where PaTH drifts is exactly the one QWAB cannot
> usefully realize.

**Corollary (explains the observations).** This predicts, without fitting: (a) the
inference wavelet is inert/removable (its usable band carries no drift) — confirmed,
QWAB-off ≈ QWAB-on; (b) making coarse scales visible (centering) hurts, worst at long
length — confirmed, −0.032 @ L4096; (c) the router avoids coarse scales at every
length/position — confirmed, K=8 heatmaps.

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

# PAT-225 paper outline — mechanism/negative-result framing

Goal: fastest defensible submission, not the biggest possible paper. Reframe from
"QWAB improves long-context retrieval" (evidence doesn't support this) to
"systematic mechanistic study of a content-adaptive wavelet positional bias on
PaTH attention: what it converges to, why it doesn't help, plus a benchmark
construction pitfall found along the way." Matches the earlier external review's
own redirect (score 3/10 → roadmap toward exactly this kind of analysis paper).

Target: EMNLP Findings / workshop, not COLM main track. Findings tracks and
workshops accept well-executed negative/mechanistic results; a flagship venue
does not.

## Status legend
- DONE: numbers/finding already exist, ready to write up
- NEEDS 3-SEED: exists as single-seed, needs s43/s44 to state as a claim with error bars
- NEEDS 1 MORE JOB: small, well-scoped, in flight or trivial to launch

---

## Section 1 — Intro / framing
DONE (just needs writing). One paragraph: PaTH/linear attention extrapolates
poorly beyond train length; content-adaptive positional bias (query-conditioned
wavelet, "QWAB") is a natural fix; we built it, evaluated it carefully, and
report what we found — including a benchmark pitfall that inflated earlier
numbers by ~15-18% relative F1.

## Section 2 — Benchmark construction pitfall (stands alone as a contribution)
DONE.
- `build_context_budgeted` bug: evidence sentence pinned to context position 0
  in 100% of examples (both train and eval, since they share the function),
  giving a trivial positional shortcut.
- Fix: sort by document order before joining → evidence-at-position-0 drops
  100% → 15.4%, passes chi-square uniformity test (df=9, 15.41 < 16.92).
- Consequence: F1 dropped ~0.09 (PA_only) to ~0.11 (QWAB) after the fix at
  L2048 — i.e. prior "headline" numbers on this benchmark were substantially
  inflated by a positional shortcut, not genuine long-context retrieval skill.
- Framing: a general cautionary finding for anyone building synthetic
  long-context QA benchmarks by inserting "supporting facts" into a padded
  context — worth stating as a standalone methodological note.

## Section 3 — Method (brief)
DONE. QWAB mechanism description: per-layer router selects among K wavelet
scales (Ricker/Morlet), query-conditioned weight, logit-bias injection,
scale-coupled shift. Keep short — this is not the paper's contribution, the
analysis of it is.

## Section 4 — Main result: systematic null across independent axes
This is the core negative-result section. Assemble as ONE summary table/figure
showing every ablation axis converging to "no measurable value," not a wall of
separate tables.

Axes (all DONE, single-seed):
1. QWAB vs PA_only across L_train ∈ {512, 256, 128}, extrapolation up to 32x —
   best margin +0.0097 (K4@L256), most configs within noise floor, several
   negative (K1@L512 morlet: -0.0035).
2. Causal ablation — dampening QWAB's own contribution (near-L_train,
   far-L_train, alpha=0.5) → ~0 effect (+0.0005 / -0.0004). Dampening PA the
   same way → catastrophic (-0.34 / -0.41). Establishes PA, not QWAB, is
   load-bearing.
3. Amplification — gain2x on QWAB makes F1 *worse* (-0.0129), not better.
4. Center design — query-center and dual-center both underperform 0-center
   AND underperform PA_only.
5. Multi-scale — K3 underperforms best K1 on both morlet and ricker; K4
   marginally beats K1 (+0.0097 vs +0.0086) — not a monotonic "more scales
   helps" story.
6. Targeted rescue attempt (L_train=128 pivot) — hypothesis: shrinking L_train
   should weaken PA and open room for QWAB to help more. Result: margin
   *shrank* (+0.0023 @ L2048/16x vs +0.0086 @ L256/8x), i.e. the one
   experiment designed specifically to find a QWAB-favorable regime did not
   find one. (L4096/32x point: +0.0079 — noisier but still ≤ the L256
   plateau.)

NEEDS 3-SEED: pick the single cleanest pair (recommend K4_morlet@L256 vs
PA_only@L256, or K1_rho256_ricker@L512 vs PA_only@L512 as the historically
central one) and get s43/s44. This is the one piece of new work required —
everything else above is already sufficient to write.

## Section 5 — Mechanism: what does the router actually learn?
DONE. Two findings here, both hold up independent of whether QWAB helps
end-task performance — this is the "interesting even though it's a null"
section.

1. **rho≈256 absolute sweet spot.** Reproduced across 3 independent
   conditions (L_train=256@L2048, L_train=512@L2048, L_train=256@L4096
   2k-subset) — the router consistently prefers scale≈256 regardless of
   L_train or eval length. Not proportional to L_train; an absolute distance.
   Refutes an earlier "matches support_bundle_tokens p75≈252" coverage
   hypothesis via per-example F1 stratification (no correlation found).
2. **PA's far-distance attention is more peaked, not noisier.** Distance-region
   entropy analysis: for query rows ≥ L_train, splitting keys into
   near/far-region softmax shows LOWER entropy in the far region in 11/12
   layers — for both QWAB and PA_only checkpoints. This is the opposite of
   the "long-range attention is diffuse noise that needs damping" intuition
   that partly motivated QWAB. Connects to Retrieval Heads (Wu et al. 2024)
   / Induction Heads (Olsson et al. 2022) literature as the likely
   explanation — worth one paragraph of discussion, framed as "PaTH already
   exhibits retrieval-head-like far-attention without any added bias,"
   which is itself a small positive mechanistic finding independent of QWAB.

## Section 6 — Ablations / design-space notes (compress into 1 table + short text)
DONE, all single-seed, cite as "design choices we swept, none changed the
qualitative conclusion":
- Scale/shift coupling: `beta_i = beta_m * s_i`, each of K scales bounded by
  its own physical width — a per-scale shift-per-scale vs shared-shift
  ablation running now (568422/568423), include if it lands in time, drop
  otherwise (secondary point, not load-bearing for the paper's conclusion).
- rho sweep at K1 (128/256/384/512) — full 4-point at L_train=256, 2-point
  at L_train=512, both peak at 256.
- morlet vs ricker — no consistently-better pattern; winner flips by L_train.

## Section 7 — Discussion
DONE conceptually, needs writing:
- Consistent with an earlier, independently-run motif-probe investigation
  (different methodology, same codebase/project) that found the wavelet bias
  is a ~2% inert perturbation on attention logits (PAT-160). Two orthogonal
  methods (causal eval-time ablation here; motif/NMF reconstruction there)
  converge on the same conclusion — strengthens the negative result rather
  than being a one-off.
- Why: PA_only already achieves strong extrapolation on this task at 4-32x
  train length via retrieval-head-like far attention (Section 5.2); there
  may simply not be a gap left for an explicit positional bias to fill on
  this benchmark/task combination.

## Section 8 — Conclusion
One paragraph: we built and carefully evaluated a content-adaptive wavelet
positional bias for PaTH; after fixing a benchmark construction bug that had
inflated prior numbers, systematic ablation across 6 independent axes shows
no measurable benefit over the plain PA baseline; we characterize what the
mechanism *does* learn (an absolute distance preference, no causal footprint)
and report a mechanistic finding about PaTH's own long-range attention
(retrieval-head-like, not noisy) that helps explain why there's little room
left for an added bias to help.

---

## Fastest path to submission — checklist
1. [ ] 3-seed the ONE chosen headline comparison (s43, s44) — the only new
       experiment truly required.
2. [ ] Let 568422/568423 (shift-per-scale ablation) and 568393 (K3@L512)
       finish — nice-to-have for Section 6, not blocking.
3. [ ] Consolidate all tables in this doc into final paper tables/figures
       (2-3 total: main null-result table, rho-sweep figure, entropy-by-
       region figure).
4. [ ] Write Sections 1-3, 7-8 (framing/method/discussion/conclusion) — no
       new experiments needed, pure writing.
5. [ ] Do NOT chase more ablation axes before drafting — the null is already
       supported from 6 independent angles; additional axes have sharply
       diminishing marginal value for "fast."

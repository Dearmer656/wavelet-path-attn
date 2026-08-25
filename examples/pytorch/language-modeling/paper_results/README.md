# Paper results — master directory of symlinks

Every subdirectory here is a category of paper result; every leaf is a symlink into the real
`hotpot_long/results_uniform/` (or a checkpoint's own `ckpt_eval_hm/`) location. Use this
directory as the single place to find "which result am I citing in the paper," instead of
re-deriving paths from Linear comments each time. See also `PAT253_CANONICAL_CHECKPOINTS.md`
for the checkpoint-provenance writeup behind the Section 1 links.

## section1_1a_gate_fixed_vs_learned/
g0-fixed ($g_0{=}1$, zero learnable params) vs. learned gate, K1 rho=256 Ricker, s42.
- `g0_fixed` — L512/L2048 done, L4096 pending (job 570601)
- `g0_learned_canonical` — canonical checkpoint (see below), L512/L2048 done, L4096 pending (job 570620)

## section1_1c_conditioning_source/
Router conditioning source: `hidden_ln` vs. default `q_minus_qcorr_meanh`, K1 rho=256 Ricker, s42.
- `hidden_ln` — all 3 lengths done (L4096=0.6655)
- `default_qcorr_canonical` — same canonical checkpoint as above, L4096 pending (job 570620)

*(Section 1b, QWAB detached from PaTH / NoPE+QWAB, has no local directory — its numbers come
from PAT-165, an older diagnostic run; see PAT-253 Linear comment for the table. Not linked here
because provenance is a Linear writeup, not a local checkpoint I've re-verified.)*

## table1_basis_ablation/
Ricker/Gaussian/Morlet/Sine, K1 rho=256, HotpotQA-Long. Ricker's original training checkpoint
(`pat234_scale_card/K1_me16_noC1_s42`) has been deleted from disk — only its results survive
here; do not expect to re-run this exact checkpoint. Morlet has all 3 seeds; Ricker/Gaussian/Sine
are single-seed (s42) per project decision (flat cross-basis differences don't need multi-seed
per [[feedback_single_seed_ok_if_effect_clear]]).

## rho_backfill_128_384/
K1 rho=128 and rho=384, Ricker, 3 seeds each — the paper's small-model single-scale rho sweep
(excludes rho=256, which lives under section1_1a/1c's canonical link instead).

## medium_k3_ricker/
Medium-model K3 [128,256,384] Ricker, 3 seeds. **s43 not yet linked — still training (job 570556)**,
add once its checkpoint/results directory exists.

## medium_scale_search_gaussamp/
Medium-model K1 gaussamp (envelope-aligned) rho sweep {128,256,384,512}, s42 only.

## xsum/
- `canonical_headline_ricker_s42` — the canonical checkpoint's XSum eval outputs
  (`ckpt_eval_hm/`, contains multiple dated CSV/PNG runs — the most recent timestamped file per
  metric is the one to cite).

## Known gaps (not yet linked — add when available)
- Medium hidden_ln (PAT-160 medium counterpart) — training in progress (job 570609).
- Table 1's XSum results per basis — not yet located/linked, only HotpotQA is linked above.
- PA-only baseline checkpoint(s) — not linked; add if the paper needs a direct path to it.

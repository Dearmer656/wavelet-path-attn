# PAT-253 canonical checkpoint paths — TMLR results

**Purpose**: pin down exactly which checkpoint each headline number in the paper comes from.
This file exists because the project accumulated multiple differently-named checkpoints that
all nominally claim to be "K1, rho=256, Ricker, s42" but have different weights (confirmed via
`md5sum model.safetensors` — different hashes despite identical `supply_model.cfg`). Always
cross-check the checkpoint path against this file before citing a number in the paper.

## Headline single-scale PaTH+QWAB (K1, rho=256, Ricker, s42, rms_joint convention)

**Canonical checkpoint (locked 2026-08-25, "most recently modified" convention):**
```
runs/pat244_dual_temp/K1_L512_me16_rho256_ricker_s42/checkpoint-15000
```
Created 2026-08-13. This is the checkpoint used for all of Section 1's necessity-ablation
comparisons (g0-fixed vs. learned gate, hidden_ln vs. q_corr conditioning source) in PAT-253.

| Length | F1 | Source job |
|---|---|---|
| L512 | 0.7675 | 570612 |
| L2048 | 0.7250 | (pre-existing) |
| L4096 | pending | 570619 |

**Rejected duplicates (same nominal config, different weights — do not use):**
- `runs/pat244_dual_temp/K1_L512_me16_rho256/checkpoint-15000` (2026-08-05, no seed/basis suffix in dir name) → L512=0.7607, L4096=0.6496. Mistakenly mixed with the canonical checkpoint's L2048 in an earlier draft of this table; corrected 2026-08-25.
- `K1_me16_noC1_s42` (checkpoint directory not yet located — extensive `find`/`grep` over `runs/` timed out repeatedly) → L512=0.7599, L2048=0.7325, L4096=0.6858. This is the checkpoint Table 1's basis-ablation Ricker row (comment 7e0dccf9 in PAT-253) is built on. **Table 1 and the Section-1 necessity ablations therefore currently cite two different checkpoints for nominally the same config** — flagged, not yet reconciled. If exact consistency across all tables is required, Table 1 needs to be re-pointed at the canonical checkpoint above (requires re-running the other 3 bases' comparison too, since Table 1's cross-basis comparison must all share one checkpoint family).
- `K1_RESULTS_MASTER.md`'s `pat225_k1sc_S1_*` table row for me16/rho256 (L512=0.7588, L2048=0.7079, L4096=0.6433) — separate legacy tracking file, pre-dates the rms_joint/K-dependent-router-mode convention lock-in (PAT-244). Out of scope per [[feedback_ignore_pre_optimal_cfg_results]] memory rule; do not cite.

## Open item

Table 1 (basis ablation) vs. Section 1 (necessity ablation) checkpoint mismatch above is unresolved.
Decide: (a) accept two different-but-same-config checkpoints for two different tables (document the
caveat in the paper), or (b) unify Table 1 onto the canonical checkpoint (costs a re-run of Morlet/
Gaussian/Sine too, since Table 1's comparison must stay within one checkpoint family for a fair
basis-only ablation).

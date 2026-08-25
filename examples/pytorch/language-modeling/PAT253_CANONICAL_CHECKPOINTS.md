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
- `K1_me16_noC1_s42` → L512=0.7599, L2048=0.7325, L4096=0.6858. This is the checkpoint Table 1's basis-ablation Ricker row (comment 7e0dccf9 in PAT-253) is built on. Located via the eval's auto-generated `README.md` model card: `base_model: runs/pat234_scale_card/K1_me16_noC1_s42/checkpoint-15000` (a PAT-234 scale-cardinality grid point). **This checkpoint has since been deleted from disk** (manually cleaned up, same pattern as the K3 Morlet checkpoint noted elsewhere in PAT-253) — it cannot be reproduced or re-verified.
  - **2026-08-25 correction**: this checkpoint's `supply_model.cfg` has no `wavelet_router_norm_mode` field, which initially looked like a pre-`rms_joint`-convention (invalid) checkpoint. Verified via `sacct`: its training job (533802) ran 2026-07-25 23:28–07-26 10:53, **before** commit `968a971f81` (2026-07-31 09:26) removed the router_logits RMS-norm that was, at the time, unconditionally applied by default (no flag needed). PAT-244's `wavelet_router_norm_mode="rms_joint"` was later added specifically to reinstate this removed default. So this checkpoint's actual forward pass already had rms_joint-equivalent normalization baked in — it is **not** a pre-convention/invalid checkpoint, just an unrelated (and now-deleted) training instance from before the flag was named. Not a reason to distrust Table 1.
  - It is still a *different* checkpoint from the Section-1 canonical one, and no longer exists to be re-verified or reused — `K1_L512_me16_rho256_ricker_s42` remains the practical canonical choice for Section 1 simply because it's the only one of the two that still exists on disk, not because the other was invalid.
- `K1_RESULTS_MASTER.md`'s `pat225_k1sc_S1_*` table row for me16/rho256 (L512=0.7588, L2048=0.7079, L4096=0.6433) — separate legacy tracking file, pre-dates the rms_joint/K-dependent-router-mode convention lock-in (PAT-244). Out of scope per [[feedback_ignore_pre_optimal_cfg_results]] memory rule; do not cite.

## Open item — resolved by deletion

Table 1's Ricker checkpoint (`pat234_scale_card/K1_me16_noC1_s42`) no longer exists on disk, so
re-pointing Table 1 at the Section-1 canonical checkpoint is not optional cleanup, it is required if
exact cross-table consistency is wanted — Table 1's current Ricker row cannot be regenerated from
its original checkpoint even if desired. Decide: (a) accept the caveat that Table 1 and Section 1 cite
two different (but same-config) checkpoints, documenting this in the paper, or (b) retrain Table 1's
4-basis comparison from scratch against the canonical checkpoint's protocol so all tables share one
checkpoint family.

"""PAT-310 W5: forget-gate attenuation in attention-weight units. The gate term G_t - G_j is added to the raw logits before
the attention scale (fla f708acd897, same as g_cumsum in Triton), so its effect on the attention weights is (G_t - G_j) * d^-1/2.
For each npz from probe_gate_retention.py: per (layer, head), median over examples of the scaled penalty to the closest
supporting-fact token; report the fraction of heads that reduce that weight by < 10x and < 2x, the median head penalty,
and the median per-token retention exp(scale * log f). Prints each file, then mean +- std (ddof=0) over the files given."""
import sys, numpy as np
SCALE = 64 ** -0.5  # GPT-2 medium head dim 1024/16
rows = []
for f in sys.argv[1:]:
    z = np.load(f); hm_ev = np.nanmedian(z['pen_ev'], 1) * SCALE; hm_logf = np.nanmean(z['logf'], 1) * SCALE
    r = dict(lt10=(hm_ev > -np.log(10)).mean() * 100, lt2=(hm_ev > -np.log(2)).mean() * 100,
             med_pen=float(np.median(hm_ev)), ret=float(np.exp(np.median(hm_logf))), n=z['logf'].shape[1],
             dist=float(np.median(z['dist_ev'][z['dist_ev'] >= 0])))
    rows.append(r)
    print(f"{f.split('/')[-1]}: n_ex={r['n']} median dist to evidence {r['dist']:.0f} | heads <10x: {r['lt10']:.1f}%  <2x: {r['lt2']:.1f}% | "
          f"median head penalty {r['med_pen']:.3f} nats | median per-token retention {r['ret']:.5f}")
if len(rows) > 1:
    print('mean+-std: ' + '  '.join(f"{k} {np.mean([r[k] for r in rows]):.3f}+-{np.std([r[k] for r in rows]):.3f}" for k in ('lt10', 'lt2', 'med_pen', 'ret')))

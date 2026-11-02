"""Summarize probe_gate_retention.py outputs: per-token retention and the forget-gate logit penalty to the evidence."""
import sys, numpy as np
INIT = {'b0': 0.0, 'b5': 5.0}  # g_proj bias at init (b0 = HF default zero bias -> f = 0.5)
for f in sys.argv[1:]:
    z = np.load(f); name = f.split('/')[-1].replace('.npz', '')
    logf, pen_ev, pen_d, dists = z['logf'], z['pen_ev'], z['pen_d'], z['dists']
    b = INIT.get(name.split('_')[0])
    init_logf = -np.log1p(np.exp(-b)) if b is not None else np.nan
    de = z['dist_ev'][z['dist_ev'] >= 0]
    hm_logf = np.nanmean(logf, 1)                      # [layer, head]
    hm_ev = np.nanmedian(pen_ev, 1)                    # [layer, head], median over examples
    print(f'\n== {name}  L={int(z["L"])}  n_ex={logf.shape[1]}  layers x heads={hm_logf.shape}  dist answer->closest evidence: median {np.median(de):.0f}')
    print(f'  per-token retention f: median over heads {np.exp(np.median(hm_logf)):.5f}  (init {np.exp(init_logf):.5f}); '
          f'10/90% heads {np.exp(np.percentile(hm_logf, 10)):.5f} / {np.exp(np.percentile(hm_logf, 90)):.5f}; '
          f'heads with f>0.999: {(np.exp(hm_logf) > 0.999).mean():.1%}, f>0.9999: {(np.exp(hm_logf) > 0.9999).mean():.1%}')
    print(f'  logit penalty to closest evidence (nats, median over ex): heads >-0.1: {(hm_ev > -0.1).mean():.1%}  >-1: {(hm_ev > -1).mean():.1%}  '
          f'>-2.3 (ret>0.1): {(hm_ev > -2.3).mean():.1%}  <-10: {(hm_ev < -10).mean():.1%}; median head {np.median(hm_ev):.2f}')
    for di, d in enumerate(dists):
        p = np.nanmedian(pen_d[:, :, di, :], 1)
        if np.all(np.isnan(p)): continue
        print(f'  d={int(d):5d}: init penalty {init_logf * d:9.2f}; learned: heads >-1 {(p > -1).mean():6.1%}  >-2.3 {(p > -2.3).mean():6.1%}  median head {np.nanmedian(p):8.2f}')
    print('  per-layer: fraction of heads with evidence penalty > -2.3 | median per-token f')
    print('   ' + ' '.join(f'{l}:{(hm_ev[l] > -2.3).mean():.2f}|{np.exp(np.median(hm_logf[l])):.4f}' for l in range(hm_ev.shape[0])))

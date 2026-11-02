"""PAT-310 W5: does the medium PaTH-FoX forget gate actually forget? Runs run_clm's own HotpotQA-Long eval (paper protocol,
evidence-first, exactly the eval's prompts) on the first --probe_n examples with batch size 1 and hooks every layer's g_proj.
The forget gate enters the attention logits as sum_{i=j+1..t} logsigmoid(g_proj(h_i)) (fla PaTHAttention, g_cumsum in the
triton kernel), so for the answer position t (last prompt token) we record per layer/head:
  * mean per-token log f over the prompt,
  * the logit penalty c_t - c_j to the closest supporting-fact token (max over evidence tokens = best-case access),
  * the penalty at fixed distances d (if d < t).
Supporting-fact spans are rebuilt with the eval's builder (same budget, respect_doc_order=False) and the rebuilt prompt
length is checked against the forward's T.
Usage: python probe_gate_retention.py --probe_out X.npz --probe_n N -- <run_clm eval args>"""
import sys, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
i = sys.argv.index('--')
pa = dict(zip(sys.argv[1:i:2], sys.argv[2:i:2])); sys.argv = ['run_clm.py'] + sys.argv[i + 1:]
OUT, N = pa['--probe_out'], int(pa['--probe_n'])
sys.path.insert(0, '/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling')
import run_clm as rc
from transformers import AutoTokenizer
argv = sys.argv
L = int(argv[argv.index('--block_size') + 1]); JSONL = argv[argv.index('--hotpot_long_jsonl') + 1]
DISTS = [d for d in (128, 512, 1024, 2048, 3072, 4000) if d < L]
rc.GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained('gpt2'); tok = rc._get_tokenizer()
nl = tok('\n', add_special_tokens=False)['input_ids']
pre = tok('Context:\n', add_special_tokens=True)['input_ids']; ansp = tok('Answer:', add_special_tokens=False)['input_ids']
EX = []
for line in open(JSONL):
    ex = json.loads(line)
    if ex['meta']['target_total_tokens'] != L: continue
    gold = ex['answer']
    if not isinstance(gold, str) or not gold.strip(): continue
    suf = tok(f"\nQuestion: {ex['question']}\n", add_special_tokens=False)['input_ids'] + ansp
    ans = tok(' ' + gold.strip(), add_special_tokens=False)['input_ids']
    b = L - len(pre) - len(suf) - len(ans) - 1
    exd = dict(ex); exd['context'] = {'title': [t for t, _ in ex['context']], 'sentences': [s for _, s in ex['context']]}
    exd['supporting_facts'] = {'title': [t for t, _ in ex['supporting_facts']], 'sent_id': [j for _, j in ex['supporting_facts']]}
    txt, ids, st = rc.build_context_budgeted(exd, b, prefer_same_title=True, min_tokens=0, respect_doc_order=False)
    if txt is None: continue
    if len(EX) >= N: break
    t2s = rc._build_title2sents(exd); sup = set()
    for t, j in rc._get_supporting_pairs(exd):
        try: sup.add(f"{t}: {t2s[t][j]}")
        except Exception: pass
    ev, pos, acc = [], 0, []
    lines = txt.split('\n')
    for n_, ln in enumerate(lines):
        li = tok(ln, add_special_tokens=False)['input_ids']
        if ln in sup: ev.extend(range(len(pre) + pos, len(pre) + pos + len(li)))
        acc.extend(li); pos += len(li)
        if n_ < len(lines) - 1: acc.extend(nl); pos += len(nl)
    EX.append(dict(ev=ev, ok=acc == list(ids), T=len(pre) + len(ids) + len(suf)))
print(f'[probe] rebuilt {len(EX)} examples, span-check ok {sum(e["ok"] for e in EX)}, with evidence {sum(bool(e["ev"]) for e in EX)}', flush=True)
S = {'k': -1, 'T': [], 'Tbad': 0, 'logf': {}, 'pen_ev': {}, 'pen_d': {}, 'dist_ev': []}
def hook_for(lid):
    def hook(mod, inp, out):
        if out.shape[1] <= 1: return  # cached decoding steps
        if lid == 0:
            S['k'] += 1; S['T'].append(int(out.shape[1]))
            e = EX[S['k']] if S['k'] < len(EX) else None
            if e is None or e['T'] != out.shape[1]: S['Tbad'] += 1
            S['dist_ev'].append(int(out.shape[1] - 1 - max(e['ev'])) if e and e['ev'] else -1)
        k = S['k']
        if k >= len(EX): return
        with torch.no_grad():
            g = F.logsigmoid(out[0].float())                 # [T,H]
            c = torch.cumsum(g, 0); t = g.shape[0] - 1
            S['logf'].setdefault(lid, []).append(g[1:].mean(0).cpu().numpy())
            ev = [j for j in EX[k]['ev'] if j < t]
            S['pen_ev'].setdefault(lid, []).append((c[t] - c[torch.tensor(ev, device=c.device)]).max(0).values.cpu().numpy() if ev else np.full(g.shape[1], np.nan))
            S['pen_d'].setdefault(lid, []).append(np.stack([(c[t] - c[t - d]).cpu().numpy() if d < t else np.full(g.shape[1], np.nan) for d in DISTS]))
    return hook
_orig_eval = rc.Trainer.evaluate
def evaluate(self, *a, **k):
    n_h = 0
    for name, m in self.model.named_modules():
        if name.endswith('g_proj') and isinstance(m, nn.Linear):
            lid = int(name.split('.h.')[1].split('.')[0]); m.register_forward_hook(hook_for(lid)); n_h += 1
    print(f'[probe] hooked {n_h} g_proj layers', flush=True); assert n_h > 0
    res = _orig_eval(self, *a, **k)
    lids = sorted(S['logf'])
    np.savez(OUT, L=L, dists=np.array(DISTS), T=np.array(S['T']), dist_ev=np.array(S['dist_ev']),
             logf=np.stack([np.stack(S['logf'][l]) for l in lids]),       # [layer, ex, head]
             pen_ev=np.stack([np.stack(S['pen_ev'][l]) for l in lids]),   # [layer, ex, head]
             pen_d=np.stack([np.stack(S['pen_d'][l]) for l in lids]))     # [layer, ex, dist, head]
    print(f'[probe] forwards={len(S["T"])} T mismatches={S["Tbad"]} saved {OUT}', flush=True)
    return res
rc.Trainer.evaluate = evaluate
rc.main()

"""PAT-310 W5: attention mass on the supporting facts at the answer position, medium FoX vs PaTH vs QWAB, doc-order evidence.
Runs run_clm's own HotpotQA-Long eval (so prompts are exactly the eval's) on the first --n examples with batch size 1, and
hooks every attention layer to read the final attention row of the last prompt token (the position that predicts the first
answer token):
  * PaTH / QWAB (path_attn, pytorch impl): the softmaxed P_wav actually used in the output einsum (Householder + QWAB bias),
    via PaTHAttention._maybe_capture_entmax_attention_stats(probs=P_wav, ...).
  * FoX: recomputed from the layer's own q/k/f projections, softmax(q.k/sqrt(d) + sum_{j=i+1..t} log f_j); checked against the
    layer's real output (probs @ v -> o_proj must reproduce the module output row).
Supporting-fact token spans come from rebuilding each example's context with the eval's builder (same budget, respect_doc_order)
and are checked against the eval's prompt length and gold answer.
Usage: python probe_evidence_attention.py --probe_out X.json --probe_n N --probe_model {fox,path} -- <run_clm eval args>"""
import sys, json, os, numpy as np, torch, torch.nn.functional as F
i = sys.argv.index('--')
pa = dict(zip(sys.argv[1:i:2], sys.argv[2:i:2])); sys.argv = ['run_clm.py'] + sys.argv[i + 1:]
OUT, N, KIND = pa['--probe_out'], int(pa['--probe_n']), pa['--probe_model']
sys.path.insert(0, '/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling')
import run_clm as rc
from transformers import AutoTokenizer
argv = sys.argv
L = int(argv[argv.index('--block_size') + 1]); JSONL = argv[argv.index('--hotpot_long_jsonl') + 1]
rc.GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained('gpt2'); tok = rc._get_tokenizer()
nl = tok('\n', add_special_tokens=False)['input_ids']
pre = tok('Context:\n', add_special_tokens=True)['input_ids']; ansp = tok('Answer:', add_special_tokens=False)['input_ids']
# ---- rebuild the eval examples in eval order (buildable rows only) with evidence token spans
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
    txt, ids, st = rc.build_context_budgeted(exd, b, prefer_same_title=True, min_tokens=0, respect_doc_order=True)
    if txt is None: continue
    if len(EX) >= N: break
    t2s = rc._build_title2sents(exd); sup = set()
    for t, j in rc._get_supporting_pairs(exd):
        try: sup.add(f"{t}: {t2s[t][j]}")
        except Exception: pass
    ev, pos, ok = [], 0, True
    lines = txt.split('\n'); acc = []
    for n_, ln in enumerate(lines):
        li = tok(ln, add_special_tokens=False)['input_ids']
        if ln in sup: ev.extend(range(len(pre) + pos, len(pre) + pos + len(li)))
        acc.extend(li); pos += len(li)
        if n_ < len(lines) - 1: acc.extend(nl); pos += len(nl)
    ok = acc == list(ids)
    EX.append(dict(ev=ev, t=len(pre) + len(ids) + len(suf) - 1, T=len(pre) + len(ids) + len(suf) + len(ans) + 1, ok=ok, gold=gold.strip(), first_ev=(min(ev) if ev else None)))
print(f'[probe] rebuilt {len(EX)} examples, span-check ok {sum(e["ok"] for e in EX)}, with evidence {sum(bool(e["ev"]) for e in EX)}', flush=True)
STATE = {'ex': -1, 'rows': [], 'cur': None, 'shape_bad': 0, 'fox_check': []}
def new_example(Tin):
    STATE['ex'] += 1
    e = EX[STATE['ex']] if STATE['ex'] < len(EX) else None
    if e is None or Tin != e['T']: STATE['shape_bad'] += 1
    STATE['cur'] = {'k': STATE['ex'], 'Tin': int(Tin), 'mass': {}, 'ctx_mass': {}}
    STATE['rows'].append(STATE['cur'])
def record(lid, row):  # row: [H, Tk] attention of the answer position
    e = EX[STATE['cur']['k']]
    ev = torch.tensor(e['ev'], dtype=torch.long, device=row.device)
    STATE['cur']['mass'][lid] = row[:, ev].sum(-1).float().cpu().tolist() if len(e['ev']) else None
    STATE['cur']['ctx_mass'][lid] = row[:, len(pre):e['t'] + 1].sum(-1).float().cpu().tolist()
_orig_eval = rc.Trainer.evaluate
def evaluate(self, *a, **k):
    model = self.model
    if KIND == 'path':
        from fla.layers import path_attn as P
        cls = P.PaTHAttention
        def cap(mod, probs, future, layer_idx):
            mod._capture_debug_tensors = False
            lid = int(mod.layer_idx if layer_idx is None else layer_idx)
            assert probs.dim() == 4 and probs.shape[0] == 1 and probs.shape[-2] == probs.shape[-1], ('need full [1,H,T,T] probs', tuple(probs.shape))
            if lid == 0: new_example(probs.shape[-1])
            t = EX[STATE['cur']['k']]['t']
            record(lid, probs[0, :, t, :].detach())
        cls._maybe_capture_entmax_attention_stats = cap
        for m in model.modules():
            if isinstance(m, cls): m._capture_debug_tensors = False
    else:
        def hook(mod, inp, kw, out):
            h = inp[0] if len(inp) else kw['hidden_states']
            lid = int(mod.layer_idx)
            if lid == 0: new_example(h.shape[1])
            t = EX[STATE['cur']['k']]['t']
            with torch.no_grad():
                q = mod.q_proj(h)[0, t].float().view(mod.num_heads, -1)            # [H,D]
                kk = mod.k_proj(h)[0, :t + 1].float().view(t + 1, mod.num_heads, -1)  # [t+1,H,D]
                f = F.logsigmoid(mod.f_proj(h)[0, :t + 1].float())                   # [t+1,H]
                c = torch.cumsum(f, 0)
                logits = torch.einsum('hd,thd->ht', q, kk) * q.shape[-1] ** -0.5 + (c[t][:, None] - c.T)
                row = torch.softmax(logits, -1)                                       # [H,t+1]
                if STATE['ex'] < 3:  # check against the layer's real output
                    vv = mod.v_proj(h)[0, :t + 1].float().view(t + 1, mod.num_heads, -1)
                    o = torch.einsum('ht,thd->hd', row, vv).reshape(-1)
                    ref = (out[0] if isinstance(out, tuple) else out)[0, t].float()
                    mine = mod.o_proj(o.to(mod.o_proj.weight.dtype)).float()
                    STATE['fox_check'].append(float((mine - ref).norm() / ref.norm()))
            record(lid, row)
        for m in model.modules():
            if m.__class__.__name__ == 'ForgettingAttention': m.register_forward_hook(hook, with_kwargs=True)
    res = _orig_eval(self, *a, **k)
    rows = STATE['rows']
    print(f'[probe] forwards={len(rows)} shape mismatches={STATE["shape_bad"]} fox output check rel.err={STATE["fox_check"][:6]}', flush=True)
    json.dump({'L': L, 'kind': KIND, 'examples': [dict(e, ev=None, n_ev=len(e['ev'])) for e in EX], 'rows': rows, 'fox_check': STATE['fox_check']}, open(OUT, 'w'))
    print('[probe] saved', OUT, flush=True)
    return res
rc.Trainer.evaluate = evaluate
rc.main()

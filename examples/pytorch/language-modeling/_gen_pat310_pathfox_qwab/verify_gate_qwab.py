"""Verify the FoX gate in the PyTorch PaTH path (QWAB kernel): inert at huge bias, Triton parity on a real PaTH-FoX checkpoint,
exact logit term at layer 0 and gradients reaching both components. Also rescales the gate probe by the attention scale."""
import copy, glob, sys, numpy as np, torch, torch.nn.functional as F
from transformers import AutoConfig, GPT2LMHeadModel
R = '/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/runs/'
QWAB = R + 'mix_medium_K1_rho256_ricker_s42/checkpoint-15000'
FOX = R + 'pat310_medium_pathfox/pathfox_b5_s42/checkpoint-15000'
dev = 'cuda'
ids = torch.randint(0, 50257, (1, 512), generator=torch.Generator().manual_seed(0)).to(dev)
def load(ck, impl, gate, mode=None):
    c = AutoConfig.from_pretrained(ck); c.path_attn_impl = impl; c.use_forget_gate = gate; c.attn_pdrop = c.resid_pdrop = c.embd_pdrop = 0.0
    if mode: c.wavelet_mode = mode
    c.attn_implementation = 'path_attn'  # AutoConfig drops it from config.json; GPT2Block reads it to build PaTH
    m, info = GPT2LMHeadModel.from_pretrained(ck, config=c, output_loading_info=True)
    miss = [k for k in info['missing_keys'] if 'g_proj' not in k]; unexp = [k for k in info['unexpected_keys'] if not (not gate and 'g_proj' in k)]
    print('load', ck.split('/')[-2], impl, 'gate', gate, 'missing', miss[:5], 'unexpected', unexp[:5], 'gates', sum(n.endswith('g_proj') for n, _ in m.named_modules()))
    assert not miss and not unexp
    return m.to(dev).eval()
def gates(m): return [x for n, x in m.named_modules() if n.endswith('g_proj')]
def set_gate(m, b, src=None):
    with torch.no_grad():
        for i, g in enumerate(gates(m)):
            if src is None: g.weight.zero_(); g.bias.fill_(b)
            else: g.load_state_dict(src[i].state_dict())
@torch.no_grad()
def out(m): return m(ids).logits.float()
print('== T1 inert: QWAB + gate(bias 40) vs QWAB, same kernel')
a = load(QWAB, 'pytorch', False); b = load(QWAB, 'pytorch', True); set_gate(b, 40.0)
print('n_gates', len(gates(b)), 'max|diff|', (out(a) - out(b)).abs().max().item())
print('== T2 parity on PaTH-FoX b5 s42 ckpt15000 (wavelet off): pytorch vs triton')
ft = load(FOX, 'triton', True, 'off'); fp = load(FOX, 'pytorch', True, 'off')
print('with gate: max|diff|', (out(ft) - out(fp)).abs().max().item(), ' effect of gate on pytorch kernel: max|diff|', end=' ')
fn = copy.deepcopy(fp); set_gate(fn, 40.0)
print((out(fp) - out(fn)).abs().max().item())
ft2 = copy.deepcopy(ft); set_gate(ft2, 40.0)
print('noise floor, gate off: max|diff|', (out(ft2) - out(fn)).abs().max().item())
print('== T3 layer-0 term: QWAB + real FoX gate minus QWAB == G_i - G_j')
b5 = load(QWAB, 'pytorch', True); set_gate(b5, 0, gates(ft))
cap = {}
h = gates(b5)[0].register_forward_hook(lambda _m, _i, o: cap.__setitem__('g', F.logsigmoid(o.float())))
out(a); out(b5); h.remove()
at = [x for n, x in a.named_modules() if type(x).__name__ == 'PaTHAttention'][0]; bt = [x for n, x in b5.named_modules() if type(x).__name__ == 'PaTHAttention'][0]
d = bt._last_logits_full - at._last_logits_full
G = torch.cumsum(cap['g'], 1).transpose(1, 2); want = G.unsqueeze(-1) - G.unsqueeze(-2)
tril = torch.tril(torch.ones(512, 512, device=dev, dtype=torch.bool))
print('max|d-want| lower-tri', (d - want)[:, :, tril].abs().max().item(), ' max|want|', want[:, :, tril].abs().max().item())
print('qwab vs qwab+gate output max|diff|', (out(a) - out(b5)).abs().max().item())
print('== T4 gradients in train mode')
b5.train(); loss = b5(ids[:, :256], labels=ids[:, :256]).loss; loss.backward()
gn = {n: p.grad.norm().item() if p.grad is not None else None for n, p in b5.named_parameters() if 'g_proj' in n or 'wavelet' in n or 'router' in n}
gg = [v for n, v in gn.items() if 'g_proj' in n]; wv = {n: v for n, v in gn.items() if 'g_proj' not in n and (v or 0) > 0}
print('loss', loss.item(), 'g_proj nonzero', sum(1 for v in gg if v and v > 0), '/', len(gg), ' qwab params nonzero', len(wv), 'e.g.', list(wv.items())[:3], ' any nan', any(torch.isnan(p.grad).any().item() for p in b5.parameters() if p.grad is not None))
print('== T5 gate probe, raw vs effective (x head_dim^-0.5 = 1/8)')
G = '/cl/work5/hongyu-s/transformers/examples/pytorch/language-modeling/hotpot_long/results_uniform/pat310_medium_pathfox/gate_probe/'
for f in sorted(glob.glob(G + '*.npz')):
    z = np.load(f); ev = np.nanmedian(z['pen_ev'], 1); lf = np.nanmean(z['logf'], 1)
    print(f.split('/')[-1], 'median penalty to evidence raw %.2f eff %.2f | heads ret>10%%: raw %.1f%% eff %.1f%% | ret>50%%: eff %.1f%% | median f raw %.5f eff %.5f' % (np.median(ev), np.median(ev) / 8, (ev > -2.3).mean() * 100, (ev / 8 > -2.3).mean() * 100, (ev / 8 > -0.693).mean() * 100, np.exp(np.median(lf)), np.exp(np.median(lf) / 8)))
    for di, dd in enumerate(z['dists']):
        p = np.nanmedian(z['pen_d'][:, :, di, :], 1)
        if not np.all(np.isnan(p)): print('   d=%d heads eff>-2.3 %.1f%%, median eff %.2f' % (dd, (p / 8 > -2.3).mean() * 100, np.nanmedian(p) / 8))

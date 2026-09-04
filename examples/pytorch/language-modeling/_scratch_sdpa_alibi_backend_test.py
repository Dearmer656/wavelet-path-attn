import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import math

def get_alibi_slopes(n_heads):
    def slopes_pow2(n):
        start = 2.0 ** (-(2.0 ** -(math.log2(n) - 3)))
        return [start * (start ** i) for i in range(n)]
    if math.log2(n_heads) % 1 == 0:
        return slopes_pow2(n_heads)
    closest = 2 ** math.floor(math.log2(n_heads))
    s = slopes_pow2(closest)
    extra = slopes_pow2(2 * closest)[0::2]
    return s + extra[: n_heads - closest]

device = "cuda"
B, H, T, D = 2, 12, 1024, 64
dtype = torch.bfloat16

q = torch.randn(B, H, T, D, device=device, dtype=dtype)
k = torch.randn(B, H, T, D, device=device, dtype=dtype)
v = torch.randn(B, H, T, D, device=device, dtype=dtype)

slopes = torch.tensor(get_alibi_slopes(H), device=device, dtype=torch.float32)
pos = torch.arange(T, device=device, dtype=torch.float32)
dist = (pos.view(T, 1) - pos.view(1, T)).clamp(min=0)  # causal distance
alibi_bias = (-slopes.view(H, 1, 1) * dist.unsqueeze(0)).to(dtype)  # [H, T, T]
alibi_bias = alibi_bias.unsqueeze(0)  # [1, H, T, T] -- broadcasts over batch

causal_mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
attn_bias = alibi_bias.masked_fill(causal_mask.view(1, 1, T, T), float("-inf"))

results = {}
for name, backend in [
    ("FLASH", SDPBackend.FLASH_ATTENTION),
    ("EFFICIENT", SDPBackend.EFFICIENT_ATTENTION),
    ("MATH", SDPBackend.MATH),
]:
    try:
        with sdpa_kernel(backend):
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, is_causal=False)
        results[name] = f"OK, out shape={tuple(out.shape)}, mean={out.float().mean().item():.4f}"
    except Exception as e:
        results[name] = f"FAILED: {type(e).__name__}: {e}"

print("=== SDPA backend + ALiBi additive attn_mask test ===")
for name, res in results.items():
    print(f"{name}: {res}")

# Also check what backend gets picked automatically (no explicit context manager)
try:
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, is_causal=False)
    print(f"AUTO (no explicit backend): OK, out shape={tuple(out.shape)}")
except Exception as e:
    print(f"AUTO (no explicit backend): FAILED: {type(e).__name__}: {e}")

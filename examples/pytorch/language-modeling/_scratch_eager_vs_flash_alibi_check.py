import torch
import torch.nn as nn
import transformers.models.gpt2.modeling_gpt2 as m

print("modeling_gpt2 module file:", m.__file__)
assert hasattr(m, "flash_attention_2_alibi_forward"), "PYTHONPATH not pointing at the edited src tree"

device = "cuda"
B, H, T, D = 2, 12, 512, 64
torch.manual_seed(0)

q = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
k = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
v = torch.randn(B, H, T, D, device=device, dtype=torch.float32)


class FakeConfig:
    pe_method = "alibi"


class FakeModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = FakeConfig()
        self.scale_attn_weights = True
        self.scale_attn_by_inverse_layer_idx = False
        self.is_cross_attention = False
        self.layer_idx = 0
        self.attn_dropout = nn.Dropout(0.0)
        max_pos = 2048
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(max_pos, max_pos, dtype=torch.bool)).view(1, 1, max_pos, max_pos),
        )


module = FakeModule().to(device)
module.eval()

with torch.no_grad():
    eager_out, _ = m.eager_attention_forward(module, q, k, v, attention_mask=None)
    flash_out, _ = m.flash_attention_2_alibi_forward(module, q, k, v, attention_mask=None)

print("eager_out shape:", tuple(eager_out.shape), "dtype:", eager_out.dtype)
print("flash_out shape:", tuple(flash_out.shape), "dtype:", flash_out.dtype)

eager_f32 = eager_out.float()
flash_f32 = flash_out.float()

abs_diff = (eager_f32 - flash_f32).abs()
rel_diff = abs_diff / (eager_f32.abs() + 1e-6)

print(f"max abs diff: {abs_diff.max().item():.6f}")
print(f"mean abs diff: {abs_diff.mean().item():.6f}")
print(f"max rel diff: {rel_diff.max().item():.6f}")
print(f"eager mean/std: {eager_f32.mean().item():.6f} / {eager_f32.std().item():.6f}")
print(f"flash mean/std: {flash_f32.mean().item():.6f} / {flash_f32.std().item():.6f}")

ok = torch.allclose(eager_f32, flash_f32, atol=2e-2, rtol=2e-2)
print("ALLCLOSE (atol=2e-2, rtol=2e-2):", ok)

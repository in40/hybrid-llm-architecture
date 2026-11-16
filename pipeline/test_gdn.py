"""
GDN reference harness (CPU-friendly, no corpus data — pure random tensors).

Tests:
  1. chunked == naive_recurrent (allclose) on random normalized q,k
  1b. strong-decay stress
  2. gradient flow: all params get finite grads
  3. bitwise determinism: two identical forward passes -> identical outputs
  4. layer smoke: shapes/finiteness
  5. long-sequence stability at realistic (slow) decay, T=2048 (Q6.2)
  6. chunk-size invariance: chunk 32/64/128 agree (Q6.3)

Run: python3 test_gdn.py
"""
import torch
from gdn import chunked, naive_recurrent, l2norm, GatedDeltaNet

torch.manual_seed(0)
B, H, T, DK, DV = 2, 4, 200, 64, 64
q = l2norm(torch.randn(B, H, T, DK))
k = l2norm(torch.randn(B, H, T, DK))
v = torch.randn(B, H, T, DV)
beta = torch.rand(B, H, T).sigmoid()
g = -torch.rand(B, H, T).abs() * 0.1          # log-decay <= 0

o_rec = naive_recurrent(q, k, v, beta, g)
o_ch = chunked(q, k, v, beta, g, chunk_size=64)
err = (o_rec - o_ch).abs().max().item()
print(f"1. recurrent vs chunked: max abs diff = {err:.3e}",
      "PASS" if err < 2e-4 else "FAIL")

g2 = -torch.rand(B, H, T).abs() * 0.5
e2 = (naive_recurrent(q, k, v, beta, g2) - chunked(q, k, v, beta, g2)).abs().max().item()
print(f"1b. strong-decay stress: max abs diff = {e2:.3e}", "PASS" if e2 < 2e-4 else "FAIL")

layer = GatedDeltaNet(d_model=256, n_heads=6)   # dk=32, dv=64
x = torch.randn(B, T, 256, requires_grad=True)
out = layer(x)
out.pow(2).mean().backward()
grads_ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())
print("2. gradient flow:", "PASS" if grads_ok and torch.isfinite(x.grad).all() else "FAIL")

with torch.no_grad():
    a = layer(x.detach()); b = layer(x.detach())
print("3. bitwise determinism:", "PASS" if torch.equal(a, b) else "FAIL")
print("4. layer smoke:", "PASS" if out.shape == (B, T, 256) and torch.isfinite(out).all() else "FAIL")

# 5. long sequence, realistic slow decay (fla init: dt~0.001-0.1, A~U(0,16))
T2 = 2048
layer2 = GatedDeltaNet(d_model=128, n_heads=4)  # dk=24, dv=48
x2 = torch.randn(1, T2, 128)
o2 = layer2(x2)
g_slow = -layer2.A_log.exp()[None, :, None] * torch.nn.functional.softplus(
    layer2.a_proj(x2) + layer2.dt_bias).transpose(1, 2)
state_probe = chunked(l2norm(layer2.q_proj(x2).view(1, T2, 4, 24).transpose(1, 2)),
                      l2norm(layer2.k_proj(x2).view(1, T2, 4, 24).transpose(1, 2)),
                      layer2.v_proj(x2).view(1, T2, 4, 48).transpose(1, 2),
                      layer2.b_proj(x2).sigmoid().transpose(1, 2), g_slow)
ok5 = torch.isfinite(o2).all().item() and torch.isfinite(state_probe).all().item()
print("5. T=2048 slow-decay stability:", "PASS" if ok5 else "FAIL")

# 6. chunk-size invariance
qs, ks = l2norm(torch.randn(1, 2, 256, 32)), l2norm(torch.randn(1, 2, 256, 32))
vs = torch.randn(1, 2, 256, 64)
bs = torch.rand(1, 2, 256).sigmoid()
gs = -torch.rand(1, 2, 256).abs() * 0.02
ref = chunked(qs, ks, vs, bs, gs, chunk_size=64)
e32 = (chunked(qs, ks, vs, bs, gs, chunk_size=32) - ref).abs().max().item()
e128 = (chunked(qs, ks, vs, bs, gs, chunk_size=128) - ref).abs().max().item()
print(f"6. chunk invariance: diff@32={e32:.2e}, diff@128={e128:.2e}",
      "PASS" if max(e32, e128) < 2e-4 else "FAIL")

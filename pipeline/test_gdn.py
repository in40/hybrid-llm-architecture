"""
GDN reference harness (CPU-friendly, no corpus data — pure random tensors).

Tests:
  1. chunked == naive_recurrent (allclose) on random q,k,v,beta,g
  2. gradient flow: all params get finite grads
  3. bitwise determinism: two identical forward passes -> identical logits
  4. layer smoke: finite outputs, shapes, backward on GatedDeltaNet module

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

# long-tail decay stress: strong decay (fast forgetting)
g2 = -torch.rand(B, H, T).abs() * 0.5
e2 = (naive_recurrent(q, k, v, beta, g2) - chunked(q, k, v, beta, g2)).abs().max().item()
print(f"1b. strong-decay stress: max abs diff = {e2:.3e}", "PASS" if e2 < 2e-4 else "FAIL")

layer = GatedDeltaNet(d_model=256, n_heads=4, d_k=64, d_v=64)
x = torch.randn(B, T, 256, requires_grad=True)
out = layer(x)
out.pow(2).mean().backward()
grads_ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())
print("2. gradient flow:", "PASS" if grads_ok and torch.isfinite(x.grad).all() else "FAIL")

with torch.no_grad():
    a = layer(x.detach()); b = layer(x.detach())
print("3. bitwise determinism:", "PASS" if torch.equal(a, b) else "FAIL")
print("4. layer smoke:", "PASS" if out.shape == (B, T, 256) and torch.isfinite(out).all() else "FAIL")

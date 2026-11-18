"""
doubling_solve validation + CPU benchmark (random tensors).

  1. doubling_solve == forward_substitution on realistic strictly-lower A
  2. chunked(solver=doubling) == chunked(substitution) == naive_recurrent
  3. gradient flow with doubling
  4. CPU timing: substitution vs doubling at realistic layer size

Run: python3 test_doubling.py
"""
import time
import torch
from gdn import (chunked, naive_recurrent, forward_substitution, doubling_solve,
                 l2norm, GatedDeltaNet)

torch.manual_seed(0)

# 1. solver equivalence on realistic A (beta<=1, l2normed k, decay<=1)
B, H, C = 4, 6, 64
k = l2norm(torch.randn(B, H, C, 96))
A = torch.tril(torch.randn(B, H, C, C) * 0.3, -1) * (k @ k.transpose(-1, -2)).abs()
r = torch.randn(B, H, C, 192)
x1 = forward_substitution(A, r)
x2 = doubling_solve(A, r)
e = (x1 - x2).abs().max().item() / x1.abs().max().item()
print(f"1. solver equivalence: rel err = {e:.2e}", "PASS" if e < 1e-5 else "FAIL")

# 2. full scan equivalence vs recurrent
q = l2norm(torch.randn(2, 4, 200, 64)); k2 = l2norm(torch.randn(2, 4, 200, 64))
v = torch.randn(2, 4, 200, 64); be = torch.rand(2, 4, 200).sigmoid()
g = -torch.rand(2, 4, 200).abs() * 0.05
ref = naive_recurrent(q, k2, v, be, g)
d_sub = (chunked(q, k2, v, be, g, solver="substitution") - ref).abs().max().item()
d_dbl = (chunked(q, k2, v, be, g, solver="doubling") - ref).abs().max().item()
print(f"2. vs recurrent: substitution {d_sub:.2e}, doubling {d_dbl:.2e}",
      "PASS" if max(d_sub, d_dbl) < 2e-4 else "FAIL")

# 3. gradients with doubling
lay = GatedDeltaNet(256, 6, solver="doubling")
x = torch.randn(2, 128, 256, requires_grad=True)
lay(x).pow(2).mean().backward()
gok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in lay.parameters())
print("3. doubling gradient flow:", "PASS" if gok and torch.isfinite(x.grad).all() else "FAIL")

# 4. CPU benchmark at realistic size (B=2, T=1024, h=6, dk=96, dv=192)
qb = l2norm(torch.randn(2, 6, 1024, 96)); kb = l2norm(torch.randn(2, 6, 1024, 96))
vb = torch.randn(2, 6, 1024, 192); bb = torch.rand(2, 6, 1024).sigmoid()
gb = -torch.rand(2, 6, 1024).abs() * 0.02
for name, solver in [("substitution", "substitution"), ("doubling", "doubling")]:
    t0 = time.perf_counter()
    for _ in range(3):
        chunked(qb, kb, vb, bb, gb, solver=solver)
    dt = (time.perf_counter() - t0) / 3
    print(f"4. {name:12s}: {dt:.2f}s per fwd (B2,T1024)")

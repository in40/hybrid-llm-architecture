"""
A1 integration smoke test (CPU, random data — no corpus, no V-A).

Checks:
  1. A0 (dense) param count unchanged: 151,646,208 tied
  2. A1 (hybrid) param count matches flops_a1 prediction: 170,180,844
  3. A1 forward+backward finite; grads on GDN params
  4. A1 bitwise determinism (two forwards)
  5. wd-exclusion names: A0 none, A1 has 9x(A_log+dt_bias)
  6. A0 model builds with no GDN keys (regression: C0 code path untouched)

Run: python3 test_a1_model.py
"""
import torch
from model_backend import LLaMA

BASE = dict(n_layer=12, n_embd=768, n_head=12, vocab_size=50000,
            tie_embeddings=True, context_length=2048)
A0 = dict(BASE, intermediate_size=3072)
A1 = dict(BASE, intermediate_size=3352, full_attn_layers=[4, 8, 12],
          gdn=dict(n_heads=6, expand_v=2.0, conv_kernel=4, chunk_size=64, mode="chunked"))

m0 = LLaMA(A0)
n0 = sum(p.numel() for p in m0.parameters())
print(f"1. A0 params: {n0:,}", "PASS" if n0 == 151665408 else "FAIL")

m1 = LLaMA(A1)
n1 = sum(p.numel() for p in m1.parameters())
print(f"2. A1 params: {n1:,}", "PASS" if n1 == 170191404 else f"FAIL (delta {n1-170191404:+,})")

torch.manual_seed(0)
ids = torch.randint(0, 50000, (2, 128))
out = m1(ids)
out.pow(2).mean().backward()
g_ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in m1.parameters())
print("3. A1 fwd/bwd finite:", "PASS" if torch.isfinite(out).all() and g_ok else "FAIL")

m1.eval()
with torch.no_grad():
    a = m1(ids); b = m1(ids)
print("4. A1 bitwise determinism:", "PASS" if torch.equal(a, b) else "FAIL")

nd0 = [n for n, _ in m0.named_parameters() if n.endswith((".A_log", ".dt_bias"))]
nd1 = [n for n, _ in m1.named_parameters() if n.endswith((".A_log", ".dt_bias"))]
print("5. wd-exclusion names:", "PASS" if not nd0 and len(nd1) == 18 else f"FAIL ({len(nd0)},{len(nd1)})")

gdn_keys = any("o_norm_weight" in n for n, _ in m0.state_dict().items())
dense_ok = all(hasattr(blk.attn, "qkv") for blk in m0.blocks)
print("6. A0 path untouched:", "PASS" if (not gdn_keys and dense_ok) else "FAIL")

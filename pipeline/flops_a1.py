"""
Analytic FLOP/param accounting for A0 (dense) vs A1 (GDN 3:1, MLP-width matched).

Model: forward MACs per token (fwd+bwd ~ 3x for matching purposes; ratio-based
gate is unaffected). Full-attention layer: 4d^2 proj + 2*T*d (scores+values)
+ 6*d*i (SwiGLU). GDN layer: q,k,v (3 d*n_h*d_k / d*n_h*d_v) + gate d*n_h*d_v
+ out (n_h*d_v*d) + a,b (2 d*n_h) + conv (4*channels) + intra-chunk ops
(~ C*(d_k+d_v)*n_h per token). Output head: d*V (tied).

Solves for MLP intermediate i' such that A1 total is within +/-0.5% of A0.
Pure stdlib — run: python3 flops_a1.py
"""

D, L, NH, T, V = 768, 12, 12, 2048, 50000
I0 = 3072
FULL_LAYERS = [4, 8, 12]           # 1-indexed positions of full attention
GDN_H, GDN_DK, GDN_DV, GDN_C = 6, 96, 192, 64   # fla-canonical: key_dim=0.75d, expand_v=2


def full_layer_macs(i):
    return 4 * D * D + 2 * T * D + 6 * D * i


def gdn_layer_macs(i):
    qk, v = GDN_H * GDN_DK, GDN_H * GDN_DV      # 0.75d, 1.5d (fla 6d^2 allocation)
    proj = D * qk + D * qk + D * v + D * v + v * D + 2 * D * GDN_H
    conv = 4 * (2 * qk + v)
    intra = GDN_C * (GDN_DK + GDN_DV) * GDN_H * 2   # A-matmul + o-matmul, x2 matmuls
    return proj + conv + intra + 6 * D * i


def a0_macs():
    return L * full_layer_macs(I0) + D * V


def a1_macs(i):
    n_full, n_gdn = len(FULL_LAYERS), L - len(FULL_LAYERS)
    return n_full * full_layer_macs(i) + n_gdn * gdn_layer_macs(i) + D * V


def params(i, gdn: bool):
    emb = D * V
    full = 4 * D * D + 3 * D * i
    qk, v = GDN_H * GDN_DK, GDN_H * GDN_DV
    gdn_l = D * qk * 2 + D * v * 2 + v * D + 2 * D * GDN_H + 4 * (2 * qk + v) + 3 * D * i + v + 2 * GDN_H
    if gdn:
        return emb + 3 * full + 9 * gdn_l
    return emb + L * full


A0 = a0_macs()
best = None
for i in range(I0, I0 + 2000, 64):
    rel = (a1_macs(i) - A0) / A0
    if best is None or abs(rel) < abs(best[1]):
        best = (i, rel)
i_star, rel = best
# also check finer grid around optimum
for i in range(max(I0, i_star - 64), i_star + 65, 8):
    r = (a1_macs(i) - A0) / A0
    if abs(r) < abs(rel):
        i_star, rel = i, r

p0, p1 = params(I0, False), params(i_star, True)
print(f"A0 fwd MACs/token: {A0/1e6:.1f}M  (params {p0/1e6:.1f}M)")
print(f"GDN saving at i=I0: {(A0 - a1_macs(I0))/A0*100:.1f}% of total")
print(f"solved MLP width i' = {i_star}  (residual FLOPs {rel*100:+.3f}%)")
print(f"A1 params: {p1/1e6:.1f}M  ({(p1-p0)/p0*100:+.1f}% vs A0)")
print(f"gate +/-0.5%: {'OK' if abs(rel) <= 0.005 else 'VIOLATED'}")

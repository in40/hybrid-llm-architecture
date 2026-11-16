"""
Gated DeltaNet (GDN) layer — reference implementation in pure PyTorch.

Canonical conventions adopted from fla (fla-org/flash-linear-attention,
layers/gated_deltanet.py + ops/gated_delta_rule/naive.py), verified 2026-11-16:
  * expand_v=2 allocation, total ~6*d^2 params: q,k: 0.75d each; v,g: 1.5d; o: 1.5d
  * causal depthwise ShortConv(k=4) + SiLU on q,k,v
  * L2-norm on q,k; readout scale = 1/sqrt(head_k)
  * decay g = -exp(A_log) * softplus(a + dt_bias); A_log=log(uniform(0,16));
    dt_bias = inv-softplus(loguniform(0.001, 0.1))  -> SLOW decay at init
  * beta = sigmoid(b); output = RMSNormGated(o, gate) -> o_proj
  * scan runs in fp32 internally (fla reference casts to float32);
    A_log/dt_bias are fp32 and excluded from weight decay (_no_weight_decay)

Two forward paths that must agree numerically (test_gdn.py):
  * naive_recurrent (ground truth)  * chunked (WY form, training path)

Deterministic by construction: matmul/einsum/loops, no atomics.
"""
import math
import torch
import torch.nn.functional as F
from torch import nn


def l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / (x.pow(2).sum(-1, keepdim=True).sqrt() + 1e-12)


def forward_substitution(L, r):
    """Solve (I+L) x = r for strictly-lower L, unit diagonal. Out-of-place
    accumulation loop: deterministic, backward-stable on CPU/CUDA
    (torch.linalg.solve_triangular backward is NaN on CPU)."""
    C = L.shape[-1]
    xs = []
    for t in range(C):
        xt = r[..., t, :]
        if t:
            xt = xt - (L[..., t, :t].unsqueeze(-1) * torch.stack(xs, dim=-2)).sum(-2)
        xs.append(xt)
    return torch.stack(xs, dim=-2)


def naive_recurrent(q, k, v, beta, g, scale=None):
    """q,k: (B,h,T,dk); v: (B,h,T,dv); beta,g: (B,h,T). fp32 internally."""
    q, k, v, beta, g = [t.float() for t in (q, k, v, beta, g)]
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    q = q * scale
    B, h, T, dk = q.shape
    S = q.new_zeros(B, h, dk, v.shape[-1])
    outs = []
    for t in range(T):
        qt, kt, vt = q[:, :, t], k[:, :, t], v[:, :, t]
        S = S * g[:, :, t, None, None].exp()
        u = beta[:, :, t, None] * (vt - torch.einsum("bhk,bhkv->bhv", kt, S))
        S = S + torch.einsum("bhk,bhv->bhkv", kt, u)
        outs.append(torch.einsum("bhk,bhkv->bhv", qt, S))
    return torch.stack(outs, dim=2)


def chunked(q, k, v, beta, g, chunk_size: int = 64, scale=None):
    """Chunked gated delta rule (training path). Same signature as naive_recurrent.

    Within a chunk (cumulative log-decay G_t <= 0 from chunk start):
      A[t,s] = beta_t (k_t.k_s) exp(G_t-G_s), s<t ; r_t = beta_t (v_t - exp(G_t) S^T k_t)
      u = (I+A)^{-1} r ; o_t = exp(G_t) S^T q_t + sum_{s<=t} (q_t.k_s) exp(G_t-G_s) u_s
      S_new = exp(G_C) S + sum_s exp(G_C-G_s) k_s u_s^T
    All exponent args are <= 0 (clamp before exp in the pairwise matrix)."""
    q, k, v, beta, g = [t.float() for t in (q, k, v, beta, g)]
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    q = q * scale
    B, h, T, dk = q.shape
    dv = v.shape[-1]
    S = q.new_zeros(B, h, dk, dv)
    outs = []
    for c0 in range(0, T, chunk_size):
        c1 = min(c0 + chunk_size, T)
        qc, kc, vc = q[:, :, c0:c1], k[:, :, c0:c1], v[:, :, c0:c1]
        bc, gc = beta[:, :, c0:c1], g[:, :, c0:c1]
        C = c1 - c0
        G = gc.cumsum(-1)
        # clamp(max=0) BEFORE exp: upper-triangle would overflow to inf
        # (masked in forward, but inf*0 -> NaN in backward)
        D = (G.unsqueeze(-1) - G.unsqueeze(-2)).clamp(max=0.0).exp()
        tril_strict = torch.tril(torch.ones(C, C, dtype=torch.bool, device=q.device), -1)
        kk = torch.einsum("bhtk,bhsk->bhts", kc, kc)
        A = (bc.unsqueeze(-1) * kk * D).masked_fill(~tril_strict, 0.0)
        pred = torch.einsum("bhtk,bhkv->bhtv", kc * G.exp().unsqueeze(-1), S)
        r = bc.unsqueeze(-1) * (vc - pred)
        u = forward_substitution(A, r)
        o_inter = torch.einsum("bhtk,bhkv->bhtv", qc * G.exp().unsqueeze(-1), S)
        tril = torch.tril(torch.ones(C, C, dtype=torch.bool, device=q.device))
        qk = (torch.einsum("bhtk,bhsk->bhts", qc, kc) * D).masked_fill(~tril, 0.0)
        outs.append(o_inter + torch.einsum("bhts,bhsv->bhtv", qk, u))
        last = G[..., -1]
        S = last.exp().unsqueeze(-1).unsqueeze(-1) * S + torch.einsum(
            "bhtk,bhtv->bhkv", kc * (last.unsqueeze(-1) - G).exp().unsqueeze(-1), u)
    return torch.cat(outs, dim=2)


class GatedDeltaNet(nn.Module):
    """Canonical GDN layer (fla conventions), pure torch. d_model divisible by 8
    recommended: key_dim = 3/4 d, value_dim = 3/2 d."""

    def __init__(self, d_model: int, n_heads: int = 6, expand_v: float = 2.0,
                 conv_kernel: int = 4, chunk_size: int = 64, mode: str = "chunked"):
        super().__init__()
        assert mode in ("chunked", "recurrent")
        self.mode, self.chunk_size = mode, chunk_size
        self.h = n_heads
        self.dk = int(0.75 * d_model / n_heads)          # key_dim = 0.75d total
        self.dv = int(self.dk * expand_v)               # value_dim = 1.5d total
        self.key_dim, self.value_dim = self.h * self.dk, self.h * self.dv
        assert self.key_dim * 4 == 3 * d_model, "0.75d must be integral (n_heads | 3d/4)"
        self.q_proj = nn.Linear(d_model, self.key_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.key_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.value_dim, bias=False)
        ch = self.key_dim * 2 + self.value_dim
        self.conv = nn.Conv1d(ch, ch, kernel_size=conv_kernel, groups=ch,
                              padding=conv_kernel - 1, bias=False)
        self.a_proj = nn.Linear(d_model, n_heads, bias=False)
        self.b_proj = nn.Linear(d_model, n_heads, bias=False)
        self.g_proj = nn.Linear(d_model, self.value_dim, bias=False)
        self.o_norm_weight = nn.Parameter(torch.ones(self.dv))
        self.o_proj = nn.Linear(self.value_dim, d_model, bias=False)
        A = torch.empty(n_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A.clamp(min=1e-6)))
        self.A_log._no_weight_decay = True
        dt = torch.exp(torch.rand(n_heads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        dt = dt.clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # inv-softplus
        self.dt_bias._no_weight_decay = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = torch.cat([self.q_proj(x), self.k_proj(x), self.v_proj(x)], dim=-1)
        qkv = F.silu(self.conv(qkv.transpose(1, 2))[..., :T].transpose(1, 2))
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = l2norm(q.view(B, T, self.h, self.dk)).transpose(1, 2)
        k = l2norm(k.view(B, T, self.h, self.dk)).transpose(1, 2)
        v = v.view(B, T, self.h, self.dv).transpose(1, 2)
        g = -self.A_log.exp()[None, :, None] * F.softplus(self.a_proj(x) + self.dt_bias).transpose(1, 2)
        beta = self.b_proj(x).sigmoid().transpose(1, 2)
        scan = chunked if self.mode == "chunked" else naive_recurrent
        o = scan(q, k, v, beta, g)                          # (B,h,T,dv) fp32
        gate = F.silu(self.g_proj(x).view(B, T, self.h, self.dv).transpose(1, 2))
        o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + 1e-5) * self.o_norm_weight * gate
        o = o.transpose(1, 2).reshape(B, T, self.value_dim)
        return self.o_proj(o)

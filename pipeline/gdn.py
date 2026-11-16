"""
Gated DeltaNet (GDN) layer — reference implementation in pure PyTorch.

Two forward paths that must agree numerically (see test_gdn.py):
  * naive_recurrent: textbook per-step recurrence (ground truth)
  * chunked:         WY-representation chunk algorithm (training path)

Deterministic by construction: matmul/einsum/triangular-solve only, no atomics.

Structure follows the Qwen3-Next GDN recipe:
  q,k,v projections -> causal depthwise conv(k=4) + SiLU -> L2-norm(q,k)
  decay  g_t = -exp(A_log) * softplus(a_t + dt_bias)      (log-decay, <= 0)
  beta beta_t = sigmoid(b_t)                              (delta-rule lr)
  gated delta rule recurrence; output gated by SiLU(gate_proj(x)) -> out_proj

Shapes: x (B,T,d) -> out (B,T,d). Per head: d_k, d_v; n_h heads.
All exponentials use non-positive arguments (differences of cumulative
log-decays), so no overflow/underflow for long sequences.
"""
import torch
import torch.nn.functional as F
from torch import nn


def l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / (x.pow(2).sum(-1, keepdim=True).sqrt() + 1e-12)


def naive_recurrent(q, k, v, beta, g):
    """Ground-truth recurrence. q,k: (B,h,T,dk); v: (B,h,T,dv); beta,g: (B,h,T).
    State S: (B,h,dk,dv).  S_t = exp(g_t) S_{t-1} + k_t u_t^T,
    u_t = beta_t (v_t - exp(g_t) S_{t-1}^T k_t);  o_t = S_t^T q_t."""
    B, h, T, dk = q.shape
    dv = v.shape[-1]
    S = q.new_zeros(B, h, dk, dv)
    outs = []
    for t in range(T):
        qt, kt, vt = q[:, :, t], k[:, :, t], v[:, :, t]
        S = S * g[:, :, t, None, None].exp()
        v_hat = vt - torch.einsum("bhk,bhkv->bhv", kt, S)
        u = beta[:, :, t, None] * v_hat
        S = S + torch.einsum("bhk,bhv->bhkv", kt, u)
        outs.append(torch.einsum("bhk,bhkv->bhv", qt, S))
    return torch.stack(outs, dim=2)  # (B,h,T,dv)


def forward_substitution(L, r):
    """Solve (I+L) x = r for strictly-lower L, unit diagonal. Pure accumulation
    loop -> deterministic and backward-stable on CPU and CUDA
    (torch.linalg.solve_triangular backward is NaN on CPU). Out-of-place for autograd."""
    C = L.shape[-1]
    xs = []
    for t in range(C):
        xt = r[..., t, :]
        if t:
            xt = xt - (L[..., t, :t].unsqueeze(-1) * torch.stack(xs, dim=-2)).sum(-2)
        xs.append(xt)
    return torch.stack(xs, dim=-2)


def chunked(q, k, v, beta, g, chunk_size: int = 64):
    """Chunked gated delta rule (training path). Same signature as naive_recurrent.

    Within a chunk with cumulative log-decay G_t (from chunk start):
      A[t,s]   = beta_t (k_t.k_s) exp(G_t-G_s), s<t   (unit lower-tri system)
      r_t      = beta_t (v_t - exp(G_t) S^T k_t)
      u        = (I+A)^{-1} r
      o_t      = exp(G_t) S^T q_t + sum_{s<=t} (q_t.k_s) exp(G_t-G_s) u_s
      S_new    = exp(G_C) S + sum_s exp(G_C-G_s) k_s u_s^T
    """
    B, h, T, dk = q.shape
    dv = v.shape[-1]
    S = q.new_zeros(B, h, dk, dv)
    outs = []
    for c0 in range(0, T, chunk_size):
        c1 = min(c0 + chunk_size, T)
        qc, kc, vc = q[:, :, c0:c1], k[:, :, c0:c1], v[:, :, c0:c1]
        bc, gc = beta[:, :, c0:c1], g[:, :, c0:c1]
        C = c1 - c0
        G = gc.cumsum(-1)                                   # (B,h,C) <= 0
        # pairwise decay D[t,s] = exp(G_t - G_s), only used for t>=s.
        # clamp(max=0) BEFORE exp: upper-triangle entries would overflow to inf
        # (masked in forward, but inf*0 -> NaN in backward).
        D = (G.unsqueeze(-1) - G.unsqueeze(-2)).clamp(max=0.0).exp()  # (B,h,C,C)
        kk = torch.einsum("bhtk,bhsk->bhts", kc, kc)
        tril_strict = torch.tril(torch.ones(C, C, dtype=torch.bool, device=q.device), -1)
        A = (bc.unsqueeze(-1) * kk * D).masked_fill(~tril_strict, 0.0)
        # r = beta * (v - exp(G) * (S^T k))
        pred = torch.einsum("bhtk,bhkv->bhtv", kc * G.exp().unsqueeze(-1), S)
        r = bc.unsqueeze(-1) * (vc - pred)
        # solve (I + A) u = r, unit lower triangular (forward substitution)
        u = forward_substitution(A, r)
        # output
        o_inter = torch.einsum("bhtk,bhkv->bhtv", qc * G.exp().unsqueeze(-1), S)
        qk = torch.einsum("bhtk,bhsk->bhts", qc, kc) * D
        tril = torch.tril(torch.ones(C, C, dtype=torch.bool, device=q.device))
        o_intra = torch.einsum("bhts,bhsv->bhtv", qk.masked_fill(~tril, 0.0), u)
        outs.append(o_inter + o_intra)
        # state update
        last = G[..., -1]                                   # (B,h)
        S = last.exp().unsqueeze(-1).unsqueeze(-1) * S + \
            torch.einsum("bhtk,bhtv->bhkv", kc * (last.unsqueeze(-1) - G).exp().unsqueeze(-1), u)
    return torch.cat(outs, dim=2)  # (B,h,T,dv)


class GatedDeltaNet(nn.Module):
    """One GDN layer replacing attention (MLP unchanged outside this module)."""

    def __init__(self, d_model: int, n_heads: int = 6, d_k: int = 128, d_v: int = 128,
                 conv_kernel: int = 4, chunk_size: int = 64, mode: str = "chunked"):
        super().__init__()
        assert mode in ("chunked", "recurrent")
        self.mode, self.chunk_size = mode, chunk_size
        self.h, self.dk, self.dv = n_heads, d_k, d_v
        qk_dim, v_dim = n_heads * d_k, n_heads * d_v
        self.q_proj = nn.Linear(d_model, qk_dim, bias=False)
        self.k_proj = nn.Linear(d_model, qk_dim, bias=False)
        self.v_proj = nn.Linear(d_model, v_dim, bias=False)
        self.conv = nn.Conv1d(qk_dim + qk_dim + v_dim, qk_dim + qk_dim + v_dim,
                              kernel_size=conv_kernel, groups=qk_dim + qk_dim + v_dim,
                              padding=conv_kernel - 1, bias=False)
        self.a_proj = nn.Linear(d_model, n_heads, bias=False)   # decay input
        self.b_proj = nn.Linear(d_model, n_heads, bias=False)   # beta input
        self.g_proj = nn.Linear(d_model, v_dim, bias=False)     # output gate
        self.o_proj = nn.Linear(v_dim, d_model, bias=False)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, n_heads + 1).float()))
        self.dt_bias = nn.Parameter(torch.zeros(n_heads))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = torch.cat([self.q_proj(x), self.k_proj(x), self.v_proj(x)], dim=-1)
        qkv = self.conv(qkv.transpose(1, 2))[..., :T].transpose(1, 2)
        qkv = F.silu(qkv)
        q, k, v = qkv.split([self.h * self.dk, self.h * self.dk, self.h * self.dv], dim=-1)
        q = l2norm(q.view(B, T, self.h, self.dk)).transpose(1, 2)
        k = l2norm(k.view(B, T, self.h, self.dk)).transpose(1, 2)
        v = v.view(B, T, self.h, self.dv).transpose(1, 2)
        g = -self.A_log.exp()[None, :, None] * F.softplus(self.a_proj(x) + self.dt_bias).transpose(1, 2)
        beta = self.b_proj(x).sigmoid().transpose(1, 2)
        scan = chunked if self.mode == "chunked" else naive_recurrent
        o = scan(q, k, v, beta, g)                          # (B,h,T,dv)
        o = o.transpose(1, 2).reshape(B, T, self.h * self.dv)
        o = o * F.silu(self.g_proj(x))
        return self.o_proj(o)

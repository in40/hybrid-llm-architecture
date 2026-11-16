"""
LLaMA-style ModelBackend for C0, implementing c0_launcher.ModelBackend.

Config-driven (n_layer, n_embd, n_head, vocab, ...). Provides:
  - train_step(batch) -> loss (forward+backward+clip+opt.step+scheduler.step)
  - eval_bpb(dataset, masked) -> BPB  (masked=True applies input token-dropout)
  - state_dict / optimizer_state / scheduler_state (+load_*) for resume

torch is imported at top; this file is syntax-checkable without torch (py_compile),
but RUNTIME needs torch on the training node.

PARAM SIZE (decided v5.0): with vocab=50K (B2) + 12L/768/3072 the model is
~152M tied (embed=lm_head) / ~190M untied. DECISION: use tie_embeddings=True
(~152M). FLOP-matched reference = 6*152M*2.5B ~= 2.28e18 (NOT the old
1.875e18 computed with naive 125M). "125M" is a class label, not the target.
Set tie_embeddings=True in config and verify count_params() before training.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from gdn import GatedDeltaNet


# ---------- building blocks ----------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm) * self.weight


class RoPE(nn.Module):
    def __init__(self, head_dim, max_seq, theta=10000.0, device=None):
        super().__init__()
        inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq, device=device).float()
        freqs = torch.outer(t, inv)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def __call__(self, seq):
        return self.cos[:seq], self.sin[:seq]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    # q,k: [B, H, T, D]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_head, rope):
        super().__init__()
        assert dim % n_head == 0
        self.n_head = n_head
        self.hd = dim // n_head
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.rope = rope

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        q = q.view(B, T, self.n_head, self.hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.hd).transpose(1, 2)
        cos, sin = self.rope(T)
        q, k = apply_rope(q, k, cos, sin)
        # causal attention (math backend honors is_causal correctly + deterministically)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, dim, n_head, ffn, rope, attn=None):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        # attn: pre-built module (GatedDeltaNet for hybrid arms) or default dense attention
        self.attn = attn if attn is not None else CausalSelfAttention(dim, n_head, rope)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class LLaMA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["n_embd"])
        rope = RoPE(cfg["n_embd"] // cfg["n_head"], cfg["context_length"])
        # hybrid support: full-attention layer positions (1-indexed); others get GDN.
        # Absent/empty -> dense baseline (C0), behavior identical to pre-hybrid code.
        full = set(cfg.get("full_attn_layers") or [])
        gdn_cfg = cfg.get("gdn") or {}
        self.blocks = nn.ModuleList()
        for i in range(cfg["n_layer"]):
            if full and (i + 1) not in full:
                attn = GatedDeltaNet(cfg["n_embd"], **gdn_cfg)
                self.blocks.append(Block(cfg["n_embd"], cfg["n_head"],
                                         cfg["intermediate_size"], rope, attn=attn))
            else:
                self.blocks.append(Block(cfg["n_embd"], cfg["n_head"],
                                         cfg["intermediate_size"], rope))
        self.final_norm = RMSNorm(cfg["n_embd"])
        self.lm_head = nn.Linear(cfg["n_embd"], cfg["vocab_size"], bias=False)
        if cfg.get("tie_embeddings", False):
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)
        # scale residual output projections by 1/sqrt(2*n_layer) (GPT-2);
        # GDN's residual output is attn.o_proj
        for b in self.blocks:
            for pn in ("attn.proj", "attn.o_proj", "ffn.down"):
                mod = b
                ok = True
                for a in pn.split("."):
                    mod = getattr(mod, a, None)
                    if mod is None:
                        ok = False
                        break
                if ok:
                    nn.init.normal_(mod.weight, mean=0.0, std=0.02 / (2 * cfg["n_layer"]) ** 0.5)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Conv1d):  # GDN short conv (std 0.02 per A1 design)
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, ids):
        x = self.tok_emb(ids)
        for b in self.blocks:
            x = b(x)
        x = self.final_norm(x)
        return self.lm_head(x)


# ---------- backend implementing c0_launcher.ModelBackend ----------
class C0Backend:
    def __init__(self, cfg, tokenizer, device="cuda"):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.device = device
        self.model = LLaMA(cfg).to(device)
        # weight-decay groups: A0 (no GDN) -> single group, byte-identical to the
        # pre-hybrid optimizer (resume-compatible). A1 -> A_log/dt_bias excluded
        # from wd (fla convention, pre-registered as new-parameter treatment).
        # Name-based: custom attrs on Parameters do not survive .to(device).
        decay, nodecay = [], []
        for n_, p in self.model.named_parameters():
            (nodecay if n_.endswith((".A_log", ".dt_bias")) else decay).append(p)
        groups = [{"params": decay, "weight_decay": cfg["training"]["weight_decay"]}]
        if nodecay:
            groups.append({"params": nodecay, "weight_decay": 0.0})
        self.optimizer = torch.optim.AdamW(
            groups,
            lr=cfg["training"]["learning_rate"],
            betas=(cfg["training"]["beta1"], cfg["training"]["beta2"]),
        )
        self.scheduler = self._make_scheduler()
        self.mask_id = tokenizer.token_to_id("<MASK>")
        self.special_ids = set()
        try:
            for tid, at in tokenizer.get_added_tokens_decoder().items():
                if getattr(at, "special", False):
                    self.special_ids.add(tid)
        except Exception:
            pass
        self.dropout = cfg.get("regularization", {}).get("input_token_dropout", 0.0)
        self.mlp_dropout = cfg.get("regularization", {}).get("mlp_dropout", 0.0)
        self.micro_batch = cfg["training"].get("micro_batch", 4)
        self._apply_mlp_dropout(self.mlp_dropout)
        self._step = 0

    def _make_scheduler(self):
        warm = self.cfg["training"]["warmup_steps"]
        total = self.cfg["training"]["max_steps"]
        def lr_at(step):
            if step < warm:
                return step / max(1, warm)
            p = (step - warm) / max(1, total - warm)
            return 0.5 * (1 + math.cos(math.pi * p))
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_at)

    def _apply_mlp_dropout(self, p):
        if p <= 0:
            return
        for m in self.model.modules():
            if isinstance(m, SwiGLU):
                m.dropout = nn.Dropout(p)
                # wrap forward to apply dropout on gate output
                orig = m.forward
                def make(mm=m, fn=orig):
                    def fwd(x):
                        return mm.down(mm.dropout(F.silu(mm.gate(x))) * mm.up(x))
                    return fwd
                m.forward = make()

    def train_step(self, batch):
        ids_all = batch["input_ids"].to(self.device)
        tgt_all = batch["labels"].to(self.device)
        n = ids_all.size(0)
        mb = min(self.micro_batch, n)
        self.optimizer.zero_grad(set_to_none=True)
        self.model.train()
        total_loss = 0.0
        for i in range(0, n, mb):
            ids = ids_all[i:i + mb]
            tgt = tgt_all[i:i + mb]
            frac = ids.size(0) / n
            inp = self._maybe_mask_inputs(ids)
            logits = self.model(inp)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), tgt.view(-1), ignore_index=-100
            )
            (loss * frac).backward()
            total_loss += loss.item() * frac
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["training"]["grad_clip"])
        self.optimizer.step()
        self.scheduler.step()
        self._step += 1
        return total_loss

    def _maybe_mask_inputs(self, ids):
        if self.dropout <= 0:
            return ids
        mask = torch.rand_like(ids, dtype=torch.float) < self.dropout
        out = ids.clone()
        out[mask] = self.mask_id
        return out

    @torch.no_grad()
    def eval_bpb(self, dataset, masked):
        self.model.eval()
        LN2 = math.log(2.0)
        tot_nats = 0.0
        tot_bytes = 0
        for seq in dataset:
            ids = torch.tensor(seq, device=self.device).unsqueeze(0)
            inp = self._maybe_mask_inputs(ids) if masked else ids
            logits = self.model(inp)
            logp = F.log_softmax(logits, dim=-1)
            # predict next token: positions 0..T-2 predict 1..T-1
            nats = -logp[0, :-1].gather(1, ids[0, 1:].unsqueeze(1)).squeeze(1)
            targets = ids[0, 1:].tolist()
            for n, t in zip(nats.tolist(), targets):
                if t in self.special_ids:
                    continue
                tot_nats += n
                tot_bytes += len(self.tokenizer.decode([t]))
        if tot_bytes == 0:
            return None
        return tot_nats / (LN2 * tot_bytes)

    # ---- state for resume ----
    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, sd):
        self.model.load_state_dict(sd)

    def optimizer_state(self):
        return self.optimizer.state_dict()

    def load_optimizer_state(self, sd):
        self.optimizer.load_state_dict(sd)

    def scheduler_state(self):
        return self.scheduler.state_dict()

    def load_scheduler_state(self, sd):
        self.scheduler.load_state_dict(sd)


# ---- helpers ----
def count_params(cfg):
    m = LLaMA(cfg)
    return sum(p.numel() for p in m.parameters())


def estimate_flops(cfg, tokens):
    n = count_params(cfg)
    return 6 * n * tokens  # standard approximation


if __name__ == "__main__":
    import sys, json
    cfg = {
        "n_layer": 12, "n_embd": 768, "n_head": 12,
        "intermediate_size": 3072, "context_length": 2048,
        "vocab_size": 50000, "tie_embeddings": False,
        "training": {"learning_rate": 3e-4, "beta1": 0.9, "beta2": 0.95,
                     "weight_decay": 0.1, "grad_clip": 1.0,
                     "warmup_steps": 500, "max_steps": 9500},
        "regularization": {"input_token_dropout": 0.0, "mlp_dropout": 0.0},
    }
    n = count_params(cfg)
    print(f"params (untied, 50K vocab): {n/1e6:.1f}M")
    cfg["tie_embeddings"] = True
    nt = count_params(cfg)
    print(f"params (tied, 50K vocab):   {nt/1e6:.1f}M  <-- DECISION (use this)")
    print(f"FLOP-matched ref @2.5B (tied): {estimate_flops(cfg, 2.5e9):.3e}  (use 2.28e18, not 1.875e18)")
    print("[DECISION v5.0] C0 proxy = tied ~152M; '125M' is a class label. "
          "Set tie_embeddings=True; FLOP reference = 2.28e18.")

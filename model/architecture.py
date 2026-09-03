"""
architecture.py — PyTorch Transformer architecture for SUB-AI.

Implements CausalSelfAttention, MLP, Transformer Block, and SUBModel
matching the exact SUBA tensor layout and C engine specifications.
Supports CUDA acceleration, FlashAttention / SDPA, and AMP (FP16/BF16).
"""

import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import SUBConfig


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention layer.

    Projects inputs to Q, K, V with a single Linear projection (no bias),
    computes scaled dot-product attention with causal masking, and projects
    back to n_embd.
    """

    def __init__(self, config: SUBConfig):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_heads = config.n_heads
        self.head_dim = config.n_embd // config.n_heads
        self.dropout_rate = config.dropout

        assert config.n_embd % config.n_heads == 0, "n_embd must be divisible by n_heads"

        # Combined Q, K, V projection: [n_embd -> 3 * n_embd] without bias
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # Output projection: [n_embd -> n_embd] without bias
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # 1. Project to QKV: [B, T, 3 * n_embd]
        qkv = self.qkv(x)

        # 2. Split into Q, K, V: each [B, T, n_embd]
        q, k, v = qkv.split(self.n_embd, dim=-1)

        # 3. Reshape into [B, n_heads, T, head_dim]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 4. Scaled dot-product attention with causal mask
        dropout_p = self.dropout_rate if self.training else 0.0
        # PyTorch F.scaled_dot_product_attention provides fast hardware-accelerated attention
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True
        )

        # 5. Transpose back and concat heads: [B, T, n_embd]
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # 6. Output projection & residual dropout
        out = self.proj(out)
        out = self.resid_dropout(out)
        return out


class MLP(nn.Module):
    """
    Feed-Forward Network (MLP) with GELU activation.
    """

    def __init__(self, config: SUBConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, config.ffn_mult * config.n_embd, bias=False)
        self.fc2 = nn.Linear(config.ffn_mult * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        # Approximate tanh formulation matches C engine and TF gelu(approximate=True)
        h = F.gelu(h, approximate="tanh")
        h = self.fc2(h)
        h = self.dropout(h)
        return h


class Block(nn.Module):
    """
    Pre-LayerNorm Transformer Block.
    """

    def __init__(self, config: SUBConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class SUBModel(nn.Module):
    """
    Full SUB-AI Transformer Language Model with tied input/output embeddings.
    """

    def __init__(self, config: SUBConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.context_len, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=1e-5)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        Args:
            idx: LongTensor of shape [B, T] with token IDs.
            targets: Optional LongTensor of shape [B, T] with next-token IDs.
        Returns:
            logits: Tensor of shape [B, T, vocab_size].
            loss: Cross entropy loss if targets provided, else None.
        """
        device = idx.device
        B, T = idx.size()

        assert T <= self.config.context_len, f"Cannot forward sequence of length {T}, context_len is {self.config.context_len}"

        # Positions 0 .. T-1
        pos = torch.arange(0, T, dtype=torch.long, device=device)

        # Token + Position Embeddings
        tok_embeddings = self.token_emb(idx)      # [B, T, n_embd]
        pos_embeddings = self.pos_emb(pos)        # [T, n_embd]
        x = self.drop(tok_embeddings + pos_embeddings)

        # Transformer Blocks
        for block in self.blocks:
            x = block(x)

        # Final LayerNorm
        x = self.ln_f(x)                          # [B, T, n_embd]

        # Weight-tied LM Head: logits = x @ token_emb.weight.T
        # token_emb.weight has shape [vocab_size, n_embd]
        logits = F.linear(x, self.token_emb.weight)  # [B, T, vocab_size]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> List[int]:
        """
        Autoregressive generation loop.
        Args:
            prompt_ids: List of token IDs.
            max_new_tokens: Number of tokens to generate.
            temperature: Sampling temperature (1.0 = standard, <1.0 = conservative, >1.0 = creative).
            top_k: Top-k filtering limit.
            device: Target device (CPU or CUDA).
        Returns:
            List of generated token IDs including prompt.
        """
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        curr_ids = list(prompt_ids)
        if not curr_ids:
            curr_ids = [0]

        idx = torch.tensor([curr_ids], dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            # Crop to maximum context length
            idx_cond = idx[:, -self.config.context_len:]
            logits, _ = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :]  # [1, vocab_size]

            if temperature <= 0.0 or top_k == 1:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-5)
                if top_k is not None and 0 < top_k < logits.size(-1):
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, next_id), dim=1)

        return idx[0].tolist()

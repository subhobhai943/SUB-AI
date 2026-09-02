"""
architecture.py — TensorFlow Transformer architecture for SUB-AI.

Implements CausalSelfAttention, MLP, Transformer Block, and SUBModel
as tf.keras Layer and Model subclasses matching the exact SUBA tensor layout.
"""

import math
import numpy as np
import tensorflow as tf
from model.config import SUBConfig


class CausalSelfAttention(tf.keras.layers.Layer):
    """
    Multi-head causal self-attention layer.

    Projects inputs to Q, K, V with a single Dense kernel, applies scaled dot-product
    attention with lower-triangular causal masking, and projects back to n_embd.
    """

    def __init__(self, config: SUBConfig, **kwargs):
        super().__init__(**kwargs)
        self.n_embd = config.n_embd
        self.n_heads = config.n_heads
        self.head_dim = config.n_embd // config.n_heads
        self.dropout_rate = config.dropout

        assert config.n_embd % config.n_heads == 0, "n_embd must be divisible by n_heads"

        self.qkv = tf.keras.layers.Dense(3 * config.n_embd, use_bias=False, name="qkv")
        self.proj = tf.keras.layers.Dense(config.n_embd, use_bias=False, name="proj")
        self.attn_dropout = tf.keras.layers.Dropout(config.dropout)
        self.resid_dropout = tf.keras.layers.Dropout(config.dropout)

    def call(self, x, training=False):
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]

        # [B, T, 3 * n_embd]
        qkv = self.qkv(x)

        # Split into Q, K, V -> each [B, T, n_embd]
        q, k, v = tf.split(qkv, 3, axis=-1)

        # Reshape to [B, n_heads, T, head_dim]
        q = tf.transpose(tf.reshape(q, [B, T, self.n_heads, self.head_dim]), [0, 2, 1, 3])
        k = tf.transpose(tf.reshape(k, [B, T, self.n_heads, self.head_dim]), [0, 2, 1, 3])
        v = tf.transpose(tf.reshape(v, [B, T, self.n_heads, self.head_dim]), [0, 2, 1, 3])

        # Scaled dot-product attention: [B, n_heads, T, T]
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = tf.matmul(q, k, transpose_b=True) * scale

        # Causal mask: lower triangular matrix
        causal_mask = tf.linalg.band_part(tf.ones((T, T), dtype=scores.dtype), -1, 0)
        mask_val = -1e9
        scores = tf.where(causal_mask == 1.0, scores, mask_val)

        attn_weights = tf.nn.softmax(scores, axis=-1)
        attn_weights = self.attn_dropout(attn_weights, training=training)

        # Attend: [B, n_heads, T, head_dim]
        out = tf.matmul(attn_weights, v)

        # Transpose back and concat heads: [B, T, n_embd]
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [B, T, self.n_embd])

        # Output projection and dropout
        out = self.proj(out)
        out = self.resid_dropout(out, training=training)
        return out


class MLP(tf.keras.layers.Layer):
    """
    Feed-Forward Network (MLP) with GELU activation.
    """

    def __init__(self, config: SUBConfig, **kwargs):
        super().__init__(**kwargs)
        self.fc1 = tf.keras.layers.Dense(config.ffn_mult * config.n_embd, use_bias=False, name="fc1")
        self.fc2 = tf.keras.layers.Dense(config.n_embd, use_bias=False, name="fc2")
        self.dropout = tf.keras.layers.Dropout(config.dropout)

    def call(self, x, training=False):
        h = self.fc1(x)
        h = tf.keras.activations.gelu(h, approximate=True)
        h = self.fc2(h)
        h = self.dropout(h, training=training)
        return h


class Block(tf.keras.layers.Layer):
    """
    Pre-LayerNorm Transformer Block.
    """

    def __init__(self, config: SUBConfig, **kwargs):
        super().__init__(**kwargs)
        self.ln_1 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="ln_1")
        self.attn = CausalSelfAttention(config, name="attn")
        self.ln_2 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="ln_2")
        self.mlp = MLP(config, name="mlp")

    def call(self, x, training=False):
        x = x + self.attn(self.ln_1(x), training=training)
        x = x + self.mlp(self.ln_2(x), training=training)
        return x


class SUBModel(tf.keras.Model):
    """
    Full SUB-AI Transformer Language Model with tied input/output embeddings.
    """

    def __init__(self, config: SUBConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        self.token_emb = tf.keras.layers.Embedding(
            config.vocab_size, config.n_embd, name="token_emb"
        )
        self.pos_emb = tf.keras.layers.Embedding(
            config.context_len, config.n_embd, name="pos_emb"
        )
        self.drop = tf.keras.layers.Dropout(config.dropout)
        self.blocks = [Block(config, name=f"block_{i}") for i in range(config.n_layers)]
        self.ln_f = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="ln_f")

    def call(self, idx, training=False):
        """
        Forward pass.
        Args:
            idx: Tensor of shape [B, T] with token IDs.
        Returns:
            logits: Tensor of shape [B, T, vocab_size].
        """
        B = tf.shape(idx)[0]
        T = tf.shape(idx)[1]

        # Token + Position Embeddings
        pos = tf.range(0, T, dtype=tf.int32)
        tok_embeddings = self.token_emb(idx)
        pos_embeddings = self.pos_emb(pos)
        x = tok_embeddings + pos_embeddings
        x = self.drop(x, training=training)

        # Transformer Blocks
        for block in self.blocks:
            x = block(x, training=training)

        # Final LayerNorm
        x = self.ln_f(x)

        # Weight-tied LM Head: logits = x @ W_emb^T
        # token_emb.embeddings has shape [vocab_size, n_embd]
        emb_weights = self.token_emb.embeddings
        logits = tf.matmul(x, emb_weights, transpose_b=True)
        return logits

    def generate(self, prompt_ids, max_new_tokens, temperature=1.0, top_k=None):
        """
        Autoregressive generation loop in NumPy.
        Args:
            prompt_ids: List or 1D array of token IDs.
            max_new_tokens: Number of tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k filtering limit.
        Returns:
            List of generated token IDs including prompt.
        """
        curr_ids = list(prompt_ids)
        if not curr_ids:
            curr_ids = [0]

        for _ in range(max_new_tokens):
            # Crop to context length
            inp = curr_ids[-self.config.context_len:]
            inp_tensor = tf.constant([inp], dtype=tf.int32)
            logits = self(inp_tensor, training=False).numpy()[0, -1, :]  # [vocab_size]

            if temperature <= 0.0 or top_k == 1:
                next_id = int(np.argmax(logits))
            else:
                logits = logits / temperature
                if top_k is not None and 0 < top_k < len(logits):
                    top_k_indices = np.argpartition(logits, -top_k)[-top_k:]
                    top_k_logits = logits[top_k_indices]
                    top_k_probs = np.exp(top_k_logits - np.max(top_k_logits))
                    top_k_probs = top_k_probs / np.sum(top_k_probs)
                    next_id = int(np.random.choice(top_k_indices, p=top_k_probs))
                else:
                    probs = np.exp(logits - np.max(logits))
                    probs = probs / np.sum(probs)
                    next_id = int(np.random.choice(len(probs), p=probs))

            curr_ids.append(next_id)

        return curr_ids

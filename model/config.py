"""
config.py — Configuration dataclass for SUB-AI Transformer architecture.

Defines hyperparameters for model dimensions, attention heads, layers,
and presets (small, medium, large).
"""

from dataclasses import dataclass


@dataclass
class SUBConfig:
    vocab_size: int = 8000
    context_len: int = 512
    n_embd: int = 256
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1
    ffn_mult: int = 4

    @classmethod
    def small(cls):
        """Preset for ~10M parameter small model."""
        return cls()

    @classmethod
    def medium(cls):
        """Preset for ~50M parameter medium model."""
        return cls(n_embd=512, n_heads=8, n_layers=8)

    @classmethod
    def large(cls):
        """Preset for ~120M parameter large model."""
        return cls(n_embd=768, n_heads=12, n_layers=12)

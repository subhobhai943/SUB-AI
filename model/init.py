"""
init.py — Custom weight initialization for SUB-AI Transformer.

Applies GPT-2 style depth-scaled truncated normal initialization
to all model weights using PyTorch initializers.
"""

import math
import torch
import torch.nn as nn
from model.config import SUBConfig


def init_weights(model: nn.Module, config: SUBConfig):
    """
    Initializes all parameters of the SUBModel according to specification:
      - Embedding + standard linear weights (qkv, fc1): TruncatedNormal(stddev=0.02)
      - Residual projection weights (proj, fc2): TruncatedNormal(stddev=0.02 / sqrt(2 * n_layers))
      - LayerNorm weight (gamma): 1.0
      - LayerNorm bias (beta) & any biases: 0.0

    Args:
        model: An instantiated SUBModel.
        config: The SUBConfig containing n_layers and hyperparameters.
    """
    resid_std = 0.02 / math.sqrt(2 * config.n_layers)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 1. LayerNorm parameters
        if "ln" in name:
            if "weight" in name:
                nn.init.ones_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        # 2. Residual projections (scaled by 1 / sqrt(2 * n_layers))
        elif "proj.weight" in name or "fc2.weight" in name:
            nn.init.trunc_normal_(
                param,
                mean=0.0,
                std=resid_std,
                a=-2.0 * resid_std,
                b=2.0 * resid_std,
            )

        # 3. Embedding and other linear projection weights (qkv, fc1)
        elif "weight" in name:
            nn.init.trunc_normal_(
                param,
                mean=0.0,
                std=0.02,
                a=-0.04,
                b=0.04,
            )

        # 4. Any biases
        elif "bias" in name:
            nn.init.zeros_(param)

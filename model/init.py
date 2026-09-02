"""
init.py — Custom weight initialization for SUB-AI Transformer.

Applies GPT-2 style depth-scaled truncated normal initialization
to all model weights using TensorFlow initializers.
"""

import math
import tensorflow as tf
from model.config import SUBConfig


def init_weights(model: tf.keras.Model, config: SUBConfig):
    """
    Initializes all trainable variables of the SUBModel according to the specification:
      - Embedding + standard linear kernels: TruncatedNormal(stddev=0.02)
      - Residual projection kernels (proj, fc2): TruncatedNormal(stddev=0.02 / sqrt(2 * n_layers))
      - LayerNorm gamma: Ones()
      - LayerNorm beta & biases: Zeros()

    Args:
        model: An instantiated SUBModel (built by passing a dummy batch).
        config: The SUBConfig containing n_layers and hyperparameters.
    """
    # Ensure model is built by doing a forward pass if not already built
    if not model.built and not model.trainable_variables:
        dummy_input = tf.zeros((1, config.context_len), dtype=tf.int32)
        model(dummy_input)

    std_normal = tf.keras.initializers.TruncatedNormal(stddev=0.02)
    resid_std = 0.02 / math.sqrt(2 * config.n_layers)
    resid_normal = tf.keras.initializers.TruncatedNormal(stddev=resid_std)
    ones_init = tf.keras.initializers.Ones()
    zeros_init = tf.keras.initializers.Zeros()

    for var in model.trainable_variables:
        name = var.name.lower()

        # LayerNorm parameters
        if "ln" in name or "layernorm" in name:
            if "gamma" in name or "scale" in name:
                var.assign(ones_init(shape=var.shape, dtype=var.dtype))
            elif "beta" in name or "offset" in name or "bias" in name:
                var.assign(zeros_init(shape=var.shape, dtype=var.dtype))
            else:
                var.assign(ones_init(shape=var.shape, dtype=var.dtype))
        # Residual projections: attn/proj and mlp/fc2
        elif "proj" in name or "fc2" in name:
            if "kernel" in name or "weight" in name:
                var.assign(resid_normal(shape=var.shape, dtype=var.dtype))
            elif "bias" in name:
                var.assign(zeros_init(shape=var.shape, dtype=var.dtype))
        # Other kernels, embeddings, and weights
        else:
            if "bias" in name:
                var.assign(zeros_init(shape=var.shape, dtype=var.dtype))
            else:
                var.assign(std_normal(shape=var.shape, dtype=var.dtype))

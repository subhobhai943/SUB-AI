"""
export.py — Export TensorFlow SUBModel checkpoint to SUBA binary format.

Reads model weights from a tf.train.Checkpoint directory or initializes a configured
model, packs tensors in float32 row-major format, and writes out the standard 256-byte
SUBA header followed by raw model weights for C engine inference.
"""

import os
import struct
import argparse
import numpy as np
import tensorflow as tf

from model.config import SUBConfig
from model.architecture import SUBModel
from model.init import init_weights

SUBA_MAGIC = 0x53554241  # ASCII 'SUBA'
SUBA_VERSION = 1


def export_model(checkpoint_path: str, out_path: str, config_preset: str = "small"):
    # 1. Config selection
    if config_preset == "small":
        config = SUBConfig.small()
    elif config_preset == "medium":
        config = SUBConfig.medium()
    else:
        config = SUBConfig.large()

    # 2. Build model
    model = SUBModel(config)
    dummy_x = tf.zeros((1, config.context_len), dtype=tf.int32)
    _ = model(dummy_x)

    # 3. Load checkpoint if provided, otherwise default to init
    if checkpoint_path and os.path.exists(checkpoint_path):
        if os.path.isdir(checkpoint_path):
            latest = tf.train.latest_checkpoint(checkpoint_path)
            if latest:
                ckpt_to_load = latest
            else:
                ckpt_to_load = checkpoint_path
        else:
            ckpt_to_load = checkpoint_path

        step_var = tf.Variable(0, dtype=tf.int64)
        ckpt = tf.train.Checkpoint(step=step_var, model=model)
        ckpt.restore(ckpt_to_load).expect_partial()
        print(f"Loaded checkpoint from: {ckpt_to_load}")
    else:
        print(f"No checkpoint found at '{checkpoint_path}'. Initializing fresh model.")
        init_weights(model, config)

    # 4. Prepare header (256 bytes = 64 uint32)
    header = [
        SUBA_MAGIC,
        SUBA_VERSION,
        config.vocab_size,
        config.context_len,
        config.n_embd,
        config.n_heads,
        config.n_layers,
    ]
    # Pad to 64 uint32 values
    header += [0] * (64 - len(header))
    header_bytes = struct.pack("<64I", *header)

    # 5. Extract weight arrays in exact order
    weight_arrays = []

    # token_emb: [vocab_size, n_embd]
    tok_emb = model.token_emb.embeddings.numpy().astype(np.float32)
    weight_arrays.append(tok_emb.flatten())

    # pos_emb: [context_len, n_embd]
    pos_emb = model.pos_emb.embeddings.numpy().astype(np.float32)
    weight_arrays.append(pos_emb.flatten())

    # Per-layer blocks
    for i, block in enumerate(model.blocks):
        # ln1.gamma, ln1.beta
        ln1_gamma = block.ln_1.gamma.numpy().astype(np.float32)
        ln1_beta = block.ln_1.beta.numpy().astype(np.float32)
        weight_arrays.append(ln1_gamma.flatten())
        weight_arrays.append(ln1_beta.flatten())

        # attn.qkv_kernel, attn.proj_kernel
        qkv_k = block.attn.qkv.kernel.numpy().astype(np.float32)
        proj_k = block.attn.proj.kernel.numpy().astype(np.float32)
        weight_arrays.append(qkv_k.flatten())
        weight_arrays.append(proj_k.flatten())

        # ln2.gamma, ln2.beta
        ln2_gamma = block.ln_2.gamma.numpy().astype(np.float32)
        ln2_beta = block.ln_2.beta.numpy().astype(np.float32)
        weight_arrays.append(ln2_gamma.flatten())
        weight_arrays.append(ln2_beta.flatten())

        # mlp.fc1_kernel, mlp.fc2_kernel
        fc1_k = block.mlp.fc1.kernel.numpy().astype(np.float32)
        fc2_k = block.mlp.fc2.kernel.numpy().astype(np.float32)
        weight_arrays.append(fc1_k.flatten())
        weight_arrays.append(fc2_k.flatten())

    # final_ln.gamma, final_ln.beta
    final_ln_gamma = model.ln_f.gamma.numpy().astype(np.float32)
    final_ln_beta = model.ln_f.beta.numpy().astype(np.float32)
    weight_arrays.append(final_ln_gamma.flatten())
    weight_arrays.append(final_ln_beta.flatten())

    # lm_head_kernel: [n_embd, vocab_size] (weight-tied = transpose of token_embedding)
    lm_head_k = tok_emb.T.astype(np.float32)
    weight_arrays.append(lm_head_k.flatten())

    # 6. Write binary file
    all_floats = np.concatenate(weight_arrays).astype(np.float32)
    total_bytes = len(header_bytes) + all_floats.nbytes

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "wb") as f:
        f.write(header_bytes)
        f.write(all_floats.tobytes())

    print(f"\n--- SUBA Export Summary ---")
    print(f"Output file:        {out_path}")
    print(f"Header size:        {len(header_bytes)} bytes")
    print(f"Total float params: {len(all_floats):,}")
    print(f"Total file size:    {total_bytes / (1024 * 1024):.2f} MB")
    print("---------------------------\n")


def main():
    parser = argparse.ArgumentParser(description="Export TensorFlow SUBModel to SUBA .bin format")
    parser.add_argument("--checkpoint", type=str, default="checkpoints", help="Path to checkpoint file or directory")
    parser.add_argument("--out", type=str, default="model.bin", help="Output .bin file path")
    parser.add_argument("--config", type=str, default="small", choices=["small", "medium", "large"], help="Config preset")
    args = parser.parse_args()

    export_model(checkpoint_path=args.checkpoint, out_path=args.out, config_preset=args.config)


if __name__ == "__main__":
    main()

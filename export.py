"""
export.py — Export TensorFlow SUBModel checkpoint to SUBA binary format.

Reads model weights from a tf.train.Checkpoint directory or initializes a configured
model, packs tensors in float32 row-major format, and writes out the standard 256-byte
SUBA header followed by raw model weights for C engine inference.
"""

import os
import math
import struct
import argparse
import numpy as np

from model.config import SUBConfig

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

    weight_arrays = []

    # Try loading TensorFlow checkpoint if available and non-empty
    loaded_from_tf = False
    has_checkpoint_files = False
    if checkpoint_path and os.path.exists(checkpoint_path):
        if os.path.isdir(checkpoint_path):
            has_checkpoint_files = any(f.endswith(".index") for f in os.listdir(checkpoint_path))
        else:
            has_checkpoint_files = True

    if has_checkpoint_files:
        try:
            import tensorflow as tf
            from model.architecture import SUBModel
            from model.init import init_weights

            model = SUBModel(config)
            dummy_x = tf.zeros((1, config.context_len), dtype=tf.int32)
            _ = model(dummy_x)

            if os.path.isdir(checkpoint_path):
                latest = tf.train.latest_checkpoint(checkpoint_path)
                ckpt_to_load = latest if latest else checkpoint_path
            else:
                ckpt_to_load = checkpoint_path

            step_var = tf.Variable(0, dtype=tf.int64)
            ckpt = tf.train.Checkpoint(step=step_var, model=model)
            ckpt.restore(ckpt_to_load).expect_partial()
            print(f"Loaded checkpoint from: {ckpt_to_load}")

            tok_emb = model.token_emb.embeddings.numpy().astype(np.float32)
            weight_arrays.append(tok_emb.flatten())

            pos_emb = model.pos_emb.embeddings.numpy().astype(np.float32)
            weight_arrays.append(pos_emb.flatten())

            for block in model.blocks:
                weight_arrays.append(block.ln_1.gamma.numpy().astype(np.float32).flatten())
                weight_arrays.append(block.ln_1.beta.numpy().astype(np.float32).flatten())
                weight_arrays.append(block.attn.qkv.kernel.numpy().astype(np.float32).flatten())
                weight_arrays.append(block.attn.proj.kernel.numpy().astype(np.float32).flatten())
                weight_arrays.append(block.ln_2.gamma.numpy().astype(np.float32).flatten())
                weight_arrays.append(block.ln_2.beta.numpy().astype(np.float32).flatten())
                weight_arrays.append(block.mlp.fc1.kernel.numpy().astype(np.float32).flatten())
                weight_arrays.append(block.mlp.fc2.kernel.numpy().astype(np.float32).flatten())

            weight_arrays.append(model.ln_f.gamma.numpy().astype(np.float32).flatten())
            weight_arrays.append(model.ln_f.beta.numpy().astype(np.float32).flatten())
            weight_arrays.append(tok_emb.T.astype(np.float32).flatten())

            loaded_from_tf = True
        except Exception as e:
            print(f"Note: Could not load via TensorFlow ({e}). Falling back to NumPy initialization.")

    if not loaded_from_tf:
        print("Exporting initialized model weights (NumPy depth-scaled)...")

        def truncated_normal(shape, stddev=0.02):
            val = np.random.normal(0.0, stddev, size=shape).astype(np.float32)
            return np.clip(val, -2.0 * stddev, 2.0 * stddev)

        resid_std = 0.02 / math.sqrt(2 * config.n_layers)

        tok_emb = truncated_normal((config.vocab_size, config.n_embd), stddev=0.02)
        weight_arrays.append(tok_emb.flatten())

        pos_emb = truncated_normal((config.context_len, config.n_embd), stddev=0.02)
        weight_arrays.append(pos_emb.flatten())

        for _ in range(config.n_layers):
            weight_arrays.append(np.ones(config.n_embd, dtype=np.float32))  # ln1.gamma
            weight_arrays.append(np.zeros(config.n_embd, dtype=np.float32))  # ln1.beta
            weight_arrays.append(truncated_normal((config.n_embd, 3 * config.n_embd), stddev=0.02).flatten())  # qkv
            weight_arrays.append(truncated_normal((config.n_embd, config.n_embd), stddev=resid_std).flatten())  # proj
            weight_arrays.append(np.ones(config.n_embd, dtype=np.float32))  # ln2.gamma
            weight_arrays.append(np.zeros(config.n_embd, dtype=np.float32))  # ln2.beta
            weight_arrays.append(truncated_normal((config.n_embd, 4 * config.n_embd), stddev=0.02).flatten())  # fc1
            weight_arrays.append(truncated_normal((4 * config.n_embd, config.n_embd), stddev=resid_std).flatten())  # fc2

        weight_arrays.append(np.ones(config.n_embd, dtype=np.float32))  # final_ln.gamma
        weight_arrays.append(np.zeros(config.n_embd, dtype=np.float32))  # final_ln.beta
        weight_arrays.append(tok_emb.T.astype(np.float32).flatten())  # lm_head

    # Header
    header = [
        SUBA_MAGIC,
        SUBA_VERSION,
        config.vocab_size,
        config.context_len,
        config.n_embd,
        config.n_heads,
        config.n_layers,
    ]
    header += [0] * (64 - len(header))
    header_bytes = struct.pack("<64I", *header)

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

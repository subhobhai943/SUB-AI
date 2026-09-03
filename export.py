"""
export.py — Export SUB-AI model checkpoint to SUBA binary format for C engine inference.

Reads model weights from a PyTorch checkpoint (.pt) or initializes configured weights,
packs tensors in float32 row-major format matching engine/loader.c and engine/matmul.c,
and writes out the standard 256-byte SUBA header followed by raw model weights.
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
    loaded_from_pt = False

    # Check if checkpoint path exists
    target_ckpt = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        if os.path.isdir(checkpoint_path):
            latest = os.path.join(checkpoint_path, "latest.pt")
            if os.path.exists(latest):
                target_ckpt = latest
            else:
                pts = [os.path.join(checkpoint_path, f) for f in os.listdir(checkpoint_path) if f.endswith(".pt")]
                if pts:
                    target_ckpt = sorted(pts)[-1]
        elif checkpoint_path.endswith(".pt"):
            target_ckpt = checkpoint_path

    if target_ckpt:
        try:
            import torch
            from model.architecture import SUBModel

            print(f"Loading checkpoint from: {target_ckpt}")
            ckpt = torch.load(target_ckpt, map_location="cpu", weights_only=False)

            model = SUBModel(config)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)
            model.eval()

            # 1. Token embedding: [vocab_size, n_embd]
            tok_emb = model.token_emb.weight.detach().cpu().numpy().astype(np.float32)
            weight_arrays.append(tok_emb.flatten())

            # 2. Position embedding: [context_len, n_embd]
            pos_emb = model.pos_emb.weight.detach().cpu().numpy().astype(np.float32)
            weight_arrays.append(pos_emb.flatten())

            # 3. Transformer blocks
            for block in model.blocks:
                # ln1 gamma & beta
                weight_arrays.append(block.ln_1.weight.detach().cpu().numpy().astype(np.float32).flatten())
                weight_arrays.append(block.ln_1.bias.detach().cpu().numpy().astype(np.float32).flatten())

                # qkv weight: PyTorch is [3*n_embd, n_embd] -> transpose to [n_embd, 3*n_embd]
                qkv_w = block.attn.qkv.weight.detach().cpu().numpy().T.astype(np.float32)
                weight_arrays.append(qkv_w.flatten())

                # proj weight: PyTorch is [n_embd, n_embd] -> transpose to [n_embd, n_embd]
                proj_w = block.attn.proj.weight.detach().cpu().numpy().T.astype(np.float32)
                weight_arrays.append(proj_w.flatten())

                # ln2 gamma & beta
                weight_arrays.append(block.ln_2.weight.detach().cpu().numpy().astype(np.float32).flatten())
                weight_arrays.append(block.ln_2.bias.detach().cpu().numpy().astype(np.float32).flatten())

                # fc1 weight: PyTorch is [4*n_embd, n_embd] -> transpose to [n_embd, 4*n_embd]
                fc1_w = block.mlp.fc1.weight.detach().cpu().numpy().T.astype(np.float32)
                weight_arrays.append(fc1_w.flatten())

                # fc2 weight: PyTorch is [n_embd, 4*n_embd] -> transpose to [4*n_embd, n_embd]
                fc2_w = block.mlp.fc2.weight.detach().cpu().numpy().T.astype(np.float32)
                weight_arrays.append(fc2_w.flatten())

            # 4. Final LayerNorm gamma & beta
            weight_arrays.append(model.ln_f.weight.detach().cpu().numpy().astype(np.float32).flatten())
            weight_arrays.append(model.ln_f.bias.detach().cpu().numpy().astype(np.float32).flatten())

            # 5. LM head (weight-tied with token_emb: x @ token_emb.T -> [n_embd, vocab_size] in C loader)
            lm_head = tok_emb.T.astype(np.float32)
            weight_arrays.append(lm_head.flatten())

            loaded_from_pt = True
            print("Successfully extracted weights from PyTorch checkpoint.")
        except Exception as e:
            print(f"Warning: Could not load via PyTorch ({e}). Falling back to NumPy initialization.")

    if not loaded_from_pt:
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

    # SUBA Binary Header: 64 x uint32 (256 bytes)
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
    parser = argparse.ArgumentParser(description="Export SUB-AI model checkpoint to SUBA .bin format")
    parser.add_argument("--checkpoint", type=str, default="checkpoints", help="Path to .pt checkpoint file or directory")
    parser.add_argument("--out", type=str, default="model.bin", help="Output .bin file path")
    parser.add_argument("--config", type=str, default="small", choices=["small", "medium", "large"], help="Config preset")
    args = parser.parse_args()

    export_model(checkpoint_path=args.checkpoint, out_path=args.out, config_preset=args.config)


if __name__ == "__main__":
    main()

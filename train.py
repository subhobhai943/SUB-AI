"""
train.py — PyTorch training pipeline for SUB-AI Transformer.

Trains the SUBModel using GPU acceleration (NVIDIA RTX 3050), AdamW optimizer with
warmup cosine learning rate schedule, gradient clipping, automatic mixed precision (AMP),
evaluation, and robust checkpointing compatible with the SUBA C inference engine.
"""

import os
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from model.config import SUBConfig
from model.architecture import SUBModel
from model.init import init_weights


def get_lr_schedule(optimizer, max_lr: float, total_steps: int, warmup_steps: int = 100, min_lr: float = None):
    """
    Returns a Cosine decay learning rate scheduler with linear warmup.
    """
    if min_lr is None:
        min_lr = max_lr / 10.0

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        # Scaled between min_lr/max_lr and 1.0
        min_ratio = min_lr / max_lr
        return min_ratio + (1.0 - min_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def get_batch(data: np.ndarray, batch_size: int, context_len: int, device: torch.device):
    """
    Sample a random batch of sequences (x, y) from numpy data array.
    """
    max_idx = len(data) - context_len - 1
    ix = np.random.randint(0, max_idx, size=(batch_size,))
    x_np = np.stack([data[i : i + context_len] for i in ix]).astype(np.int64)
    y_np = np.stack([data[i + 1 : i + 1 + context_len] for i in ix]).astype(np.int64)
    x = torch.from_numpy(x_np).to(device, non_blocking=True)
    y = torch.from_numpy(y_np).to(device, non_blocking=True)
    return x, y


@torch.no_grad()
def evaluate(
    model: SUBModel,
    val_data: np.ndarray,
    batch_size: int,
    context_len: int,
    device: torch.device,
    eval_batches: int = 20,
) -> float:
    """
    Compute average validation loss across several random batches.
    """
    model.eval()
    losses = []
    for _ in range(eval_batches):
        x, y = get_batch(val_data, batch_size, context_len, device)
        _, loss = model(x, targets=y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main():
    parser = argparse.ArgumentParser(description="Train SUB-AI Transformer model in PyTorch (GPU accelerated)")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing train.npy and val.npy")
    parser.add_argument("--config", type=str, default="small", choices=["small", "medium", "large"], help="Config preset")
    parser.add_argument("--steps", type=int, default=1000, help="Total training steps")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per step")
    parser.add_argument("--lr", type=float, default=6e-4, help="Peak learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Linear warmup steps")
    parser.add_argument("--eval_interval", type=int, default=500, help="Evaluation interval in steps")
    parser.add_argument("--save_interval", type=int, default=500, help="Checkpoint save interval in steps")
    parser.add_argument("--log_interval", type=int, default=100, help="Log interval in steps")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--no_amp", action="store_true", help="Disable Automatic Mixed Precision (AMP)")
    args = parser.parse_args()

    # 1. Hardware & Device Selection
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"\n[GPU Detected] Using {device_name} ({vram_gb:.2f} GB VRAM)")
        # Enable Ampere Tensor Core optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        use_amp = not args.no_amp
    else:
        device = torch.device("cpu")
        print("\n[CPU Mode] No CUDA GPU detected, falling back to CPU.")
        use_amp = False

    # 2. Load Data
    train_path = os.path.join(args.data_dir, "train.npy")
    val_path = os.path.join(args.data_dir, "val.npy")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(
            f"Dataset files not found in {args.data_dir}. Run 'python data/prepare.py' first."
        )

    train_data = np.load(train_path)
    val_data = np.load(val_path)
    print(f"Loaded dataset: {len(train_data):,} train tokens, {len(val_data):,} val tokens")

    # 3. Model & Configuration
    if args.config == "small":
        config = SUBConfig.small()
    elif args.config == "medium":
        config = SUBConfig.medium()
    else:
        config = SUBConfig.large()

    print(f"Model config: {args.config} (embd={config.n_embd}, heads={config.n_heads}, layers={config.n_layers}, ctx={config.context_len})")

    model = SUBModel(config)
    init_weights(model, config)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Initialized SUBModel with {total_params:,} parameters.")

    # 4. Optimizer & Schedulers
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
    )
    scheduler = get_lr_schedule(
        optimizer,
        max_lr=args.lr,
        total_steps=args.steps,
        warmup_steps=args.warmup_steps,
        min_lr=args.lr / 10.0,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # 5. Checkpoint Restore
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    latest_ckpt_path = os.path.join(args.checkpoint_dir, "latest.pt")
    start_step = 0

    if args.resume and os.path.exists(latest_ckpt_path):
        print(f"Resuming from {latest_ckpt_path}...")
        ckpt = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_step = ckpt.get("step", 0)
        print(f"Successfully resumed at step {start_step}")

    # 6. Training Loop
    print(f"\nStarting training for {args.steps} steps (AMP: {'ON' if use_amp else 'OFF'})...")
    t0 = time.time()
    accum_loss = 0.0
    log_count = 0

    model.train()
    for step in range(start_step, args.steps):
        x_batch, y_batch = get_batch(train_data, args.batch_size, config.context_len, device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=use_amp, dtype=torch.float16):
            _, loss = model(x_batch, targets=y_batch)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        accum_loss += loss.item()
        log_count += 1

        # Periodic logging
        if (step + 1) % args.log_interval == 0 or (step + 1) == args.steps:
            avg_train_loss = accum_loss / log_count
            current_lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            tok_per_sec = (log_count * args.batch_size * config.context_len) / max(elapsed, 1e-6)
            print(
                f"Step {step + 1:5d}/{args.steps:5d} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Speed: {tok_per_sec:,.0f} tok/s"
            )
            accum_loss = 0.0
            log_count = 0
            t0 = time.time()

        # Periodic evaluation
        if (step + 1) % args.eval_interval == 0 or (step + 1) == args.steps:
            val_loss = evaluate(model, val_data, args.batch_size, config.context_len, device)
            print(f"  --> Evaluation @ Step {step + 1}: Val Loss = {val_loss:.4f}")

        # Periodic checkpointing
        if (step + 1) % args.save_interval == 0 or (step + 1) == args.steps:
            ckpt_dict = {
                "step": step + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": {
                    "vocab_size": config.vocab_size,
                    "context_len": config.context_len,
                    "n_embd": config.n_embd,
                    "n_heads": config.n_heads,
                    "n_layers": config.n_layers,
                    "preset": args.config,
                },
                "loss": loss.item(),
            }
            step_ckpt = os.path.join(args.checkpoint_dir, f"sub_ai_step_{step + 1}.pt")
            torch.save(ckpt_dict, step_ckpt)
            torch.save(ckpt_dict, latest_ckpt_path)
            print(f"  --> Checkpoint saved: {step_ckpt}")

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()

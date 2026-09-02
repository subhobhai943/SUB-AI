"""
train.py — TensorFlow training pipeline for SUB-AI Transformer.

Trains the SUBModel using custom tf.GradientTape loop, Adam optimizer with
warmup cosine learning rate schedule, gradient clipping, evaluation, and
robust checkpointing via tf.train.CheckpointManager.
"""

import os
import time
import math
import argparse
import numpy as np
import tensorflow as tf

from model.config import SUBConfig
from model.architecture import SUBModel
from model.init import init_weights


class WarmupCosineDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Cosine decay learning rate schedule with linear warmup.
    """

    def __init__(self, max_lr: float, total_steps: int, warmup_steps: int = 100, min_lr: float = None):
        super().__init__()
        self.max_lr = float(max_lr)
        self.total_steps = float(total_steps)
        self.warmup_steps = float(warmup_steps)
        self.min_lr = float(min_lr if min_lr is not None else max_lr / 10.0)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.max_lr * (step / tf.maximum(self.warmup_steps, 1.0))

        progress = (step - self.warmup_steps) / tf.maximum(self.total_steps - self.warmup_steps, 1.0)
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        cosine_lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + tf.cos(math.pi * progress))

        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "max_lr": self.max_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr": self.min_lr,
        }


def get_batch(data: np.ndarray, batch_size: int, context_len: int):
    """
    Sample random batch of sequences (x, y) from data array.
    """
    max_idx = len(data) - context_len - 1
    ix = np.random.randint(0, max_idx, size=(batch_size,))
    x = np.stack([data[i : i + context_len] for i in ix]).astype(np.int32)
    y = np.stack([data[i + 1 : i + 1 + context_len] for i in ix]).astype(np.int32)
    return tf.constant(x, dtype=tf.int32), tf.constant(y, dtype=tf.int32)


def evaluate(model: SUBModel, val_data: np.ndarray, batch_size: int, context_len: int, eval_batches: int = 20):
    """
    Compute average validation loss across several random batches.
    """
    losses = []
    for _ in range(eval_batches):
        x, y = get_batch(val_data, batch_size, context_len)
        logits = model(x, training=False)
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)
        )
        losses.append(loss.numpy())
    return float(np.mean(losses))


def main():
    parser = argparse.ArgumentParser(description="Train SUB-AI Transformer model in TensorFlow")
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
    args = parser.parse_args()

    # 1. Load Data
    train_path = os.path.join(args.data_dir, "train.npy")
    val_path = os.path.join(args.data_dir, "val.npy")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(
            f"Dataset files not found in {args.data_dir}. Run 'python data/prepare.py' first."
        )

    train_data = np.load(train_path)
    val_data = np.load(val_path)
    print(f"Loaded dataset: {len(train_data):,} train tokens, {len(val_data):,} val tokens")

    # 2. Config & Model
    if args.config == "small":
        config = SUBConfig.small()
    elif args.config == "medium":
        config = SUBConfig.medium()
    else:
        config = SUBConfig.large()

    print(f"Model config: {args.config} (embd={config.n_embd}, heads={config.n_heads}, layers={config.n_layers}, ctx={config.context_len})")

    model = SUBModel(config)
    # Build model by passing a dummy batch
    dummy_x = tf.zeros((1, config.context_len), dtype=tf.int32)
    _ = model(dummy_x)

    # Initialize weights with depth scaling
    init_weights(model, config)
    print(f"Initialized {len(model.trainable_variables)} trainable variables with depth scaling.")

    # 3. Optimizer & LR Schedule
    lr_schedule = WarmupCosineDecaySchedule(
        max_lr=args.lr,
        total_steps=args.steps,
        warmup_steps=args.warmup_steps,
        min_lr=args.lr / 10.0,
    )
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        beta_1=0.9,
        beta_2=0.95,
        epsilon=1e-8,
    )

    # 4. Checkpoints
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    step_var = tf.Variable(0, dtype=tf.int64)
    checkpoint = tf.train.Checkpoint(step=step_var, optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(checkpoint, args.checkpoint_dir, max_to_keep=5)

    start_step = 0
    if args.resume and manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        start_step = int(step_var.numpy())
        print(f"Resumed from {manager.latest_checkpoint} at step {start_step}")

    # 5. Training Step (compiled with tf.function)
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)
            )
        grads = tape.gradient(loss, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    # 6. Training Loop
    print(f"\nStarting training for {args.steps} steps...")
    t0 = time.time()
    accum_loss = 0.0
    log_count = 0

    for step in range(start_step, args.steps):
        x_batch, y_batch = get_batch(train_data, args.batch_size, config.context_len)
        step_loss = train_step(x_batch, y_batch)
        accum_loss += float(step_loss.numpy())
        log_count += 1
        step_var.assign(step + 1)

        # Log training loss
        if (step + 1) % args.log_interval == 0 or (step + 1) == args.steps:
            avg_train_loss = accum_loss / log_count
            current_lr = lr_schedule(tf.constant(step, dtype=tf.float32)).numpy()
            elapsed = time.time() - t0
            tok_per_sec = (log_count * args.batch_size * config.context_len) / max(elapsed, 1e-6)
            print(f"Step {step + 1:5d}/{args.steps:5d} | Train Loss: {avg_train_loss:.4f} | LR: {current_lr:.2e} | Speed: {tok_per_sec:.0f} tok/s")
            accum_loss = 0.0
            log_count = 0
            t0 = time.time()

        # Evaluate validation loss
        if (step + 1) % args.eval_interval == 0 or (step + 1) == args.steps:
            val_loss = evaluate(model, val_data, args.batch_size, config.context_len)
            print(f"  --> Evaluation @ Step {step + 1}: Val Loss = {val_loss:.4f}")

        # Save checkpoint
        if (step + 1) % args.save_interval == 0 or (step + 1) == args.steps:
            save_path = manager.save()
            print(f"  --> Checkpoint saved: {save_path}")

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()

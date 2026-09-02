"""
prepare.py — Data preparation pipeline for SUB-AI.

Downloads TinyShakespeare corpus, trains ByteLevelBPETokenizer (vocab_size=8000),
saves tokenizer to data/tokenizer.json, tokenizes the corpus, splits into 90% train
and 10% validation sets, and saves them as data/train.npy and data/val.npy (uint16).
"""

import os
import sys
import argparse
import requests
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenizer.tokenizer import ByteLevelBPETokenizer

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def prepare_data(data_dir: str = "data", vocab_size: int = 8000):
    os.makedirs(data_dir, exist_ok=True)
    input_file = os.path.join(data_dir, "input.txt")
    tokenizer_file = os.path.join(data_dir, "tokenizer.json")
    train_file = os.path.join(data_dir, "train.npy")
    val_file = os.path.join(data_dir, "val.npy")

    # 1. Download corpus if not present
    if not os.path.exists(input_file):
        print(f"Downloading TinyShakespeare dataset from {DATA_URL}...")
        resp = requests.get(DATA_URL, timeout=30)
        resp.raise_for_status()
        with open(input_file, "w", encoding="utf-8") as f:
            f.write(resp.text)
        print(f"Saved raw dataset to {input_file} ({len(resp.text)} chars)")
    else:
        print(f"Dataset already exists at {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # 2. Train Tokenizer
    print(f"Training ByteLevelBPETokenizer with vocab_size={vocab_size}...")
    tokenizer = ByteLevelBPETokenizer(vocab_size=vocab_size)
    tokenizer.train(raw_text, vocab_size=vocab_size, verbose=True)
    tokenizer.save(tokenizer_file)
    print(f"Saved tokenizer to {tokenizer_file}")

    # 3. Tokenize corpus
    print("Tokenizing entire corpus...")
    tokens = tokenizer.encode(raw_text)
    total_tokens = len(tokens)
    print(f"Total tokens produced: {total_tokens:,}")

    # 4. Train / Val split (90% train, 10% val)
    n_train = int(0.9 * total_tokens)
    train_tokens = np.array(tokens[:n_train], dtype=np.uint16)
    val_tokens = np.array(tokens[n_train:], dtype=np.uint16)

    # 5. Save .npy files
    np.save(train_file, train_tokens)
    np.save(val_file, val_tokens)

    print("\n--- Data Preparation Summary ---")
    print(f"Vocab size:       {tokenizer.vocab_size:,}")
    print(f"Total tokens:     {total_tokens:,}")
    print(f"Train set tokens: {len(train_tokens):,} (saved to {train_file})")
    print(f"Val set tokens:   {len(val_tokens):,} (saved to {val_file})")
    print("--------------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset and train tokenizer for SUB-AI")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory to store datasets and tokenizer")
    parser.add_argument("--vocab_size", type=int, default=8000, help="Target vocabulary size")
    args = parser.parse_args()

    prepare_data(data_dir=args.data_dir, vocab_size=args.vocab_size)

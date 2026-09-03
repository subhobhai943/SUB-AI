"""
prepare.py — Data preparation pipeline for SUB-AI.

Downloads TinyStories (or TinyShakespeare), trains ByteLevelBPETokenizer with
whitespace preservation (vocab_size=8000), saves tokenizer to data/tokenizer.json,
tokenizes the corpus, splits into 90% train and 10% validation sets, and saves them
as data/train.npy and data/val.npy (uint16).
"""

import os
import sys
import argparse
import urllib.request
import requests
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenizer.tokenizer import ByteLevelBPETokenizer

TINY_STORIES_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_corpus(dataset_name: str, input_file: str, max_bytes: int = 15_000_000):
    """
    Downloads raw text corpus. For TinyStories, downloads a fast ~15 MB slice
    suitable for training small models on consumer GPUs.
    """
    if os.path.exists(input_file):
        print(f"Dataset already exists at {input_file} ({os.path.getsize(input_file):,} bytes)")
        with open(input_file, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    print(f"Downloading {dataset_name} dataset...")
    if dataset_name.lower() == "tinystories":
        req = urllib.request.Request(
            TINY_STORIES_URL,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Range": f"bytes=0-{max_bytes}",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()

        # Decode safely and truncate to last newline to avoid partial sentences
        text = data.decode("utf-8", errors="replace")
        last_nl = text.rfind("\n")
        if last_nl > 0:
            text = text[:last_nl]

        with open(input_file, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved {dataset_name} dataset to {input_file} ({len(text):,} chars, ~{len(text)/(1024*1024):.1f} MB)")
        return text
    else:
        resp = requests.get(SHAKESPEARE_URL, timeout=30)
        resp.raise_for_status()
        with open(input_file, "w", encoding="utf-8") as f:
            f.write(resp.text)
        print(f"Saved {dataset_name} dataset to {input_file} ({len(resp.text):,} chars)")
        return resp.text


def prepare_data(data_dir: str = "data", dataset: str = "tinystories", vocab_size: int = 8000, max_bytes: int = 15_000_000):
    os.makedirs(data_dir, exist_ok=True)
    input_file = os.path.join(data_dir, f"{dataset}.txt")
    tokenizer_file = os.path.join(data_dir, "tokenizer.json")
    train_file = os.path.join(data_dir, "train.npy")
    val_file = os.path.join(data_dir, "val.npy")

    # 1. Download / Load corpus
    raw_text = download_corpus(dataset, input_file, max_bytes=max_bytes)

    # 2. Train Tokenizer
    print(f"\nTraining ByteLevelBPETokenizer with vocab_size={vocab_size} on {len(raw_text):,} characters...")
    tokenizer = ByteLevelBPETokenizer(vocab_size=vocab_size)
    tokenizer.train(raw_text, vocab_size=vocab_size, verbose=True)
    tokenizer.save(tokenizer_file)
    print(f"Saved tokenizer to {tokenizer_file}")

    # 3. Tokenize corpus
    print("\nTokenizing entire corpus...")
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
    print(f"Dataset:          {dataset}")
    print(f"Vocab size:       {tokenizer.vocab_size:,}")
    print(f"Total tokens:     {total_tokens:,}")
    print(f"Train set tokens: {len(train_tokens):,} (saved to {train_file})")
    print(f"Val set tokens:   {len(val_tokens):,} (saved to {val_file})")
    print("--------------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset and train tokenizer for SUB-AI")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory to store datasets and tokenizer")
    parser.add_argument("--dataset", type=str, default="tinystories", choices=["tinystories", "tinyshakespeare"], help="Dataset to use")
    parser.add_argument("--vocab_size", type=int, default=8000, help="Target vocabulary size")
    parser.add_argument("--max_bytes", type=int, default=15_000_000, help="Max bytes to download for TinyStories slice")
    args = parser.parse_args()

    prepare_data(data_dir=args.data_dir, dataset=args.dataset, vocab_size=args.vocab_size, max_bytes=args.max_bytes)

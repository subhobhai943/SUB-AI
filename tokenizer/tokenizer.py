"""
tokenizer.py — Byte-Level Byte-Pair Encoding (BPE) Tokenizer from scratch.

Provides ByteLevelBPETokenizer without external dependencies.
Trains merges from raw text, encodes strings to token IDs, decodes IDs to text,
and saves/loads JSON format compatible with both Python and the C engine.
"""

import json
from typing import Dict, List, Tuple


class ByteLevelBPETokenizer:
    """
    Byte-Level BPE Tokenizer built completely from scratch.

    Initial vocabulary is 256 byte values (0-255). Additional tokens
    (256 to vocab_size - 1) are iteratively constructed by finding and merging
    the most frequent adjacent token pairs.
    """

    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.merges: List[Tuple[int, int]] = []
        self.bpe_ranks: Dict[Tuple[int, int], int] = {}

    def _get_pair_counts(self, token_list: List[int]) -> Dict[Tuple[int, int], int]:
        counts: Dict[Tuple[int, int], int] = {}
        for i in range(len(token_list) - 1):
            pair = (token_list[i], token_list[i + 1])
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def train(self, text: str, vocab_size: int = None, verbose: bool = False):
        """
        Train BPE tokenizer on text until target vocab_size is reached.
        """
        if vocab_size is not None:
            self.vocab_size = vocab_size

        tokens = list(text.encode("utf-8"))
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = []
        self.bpe_ranks = {}

        num_merges_target = self.vocab_size - 256
        if verbose:
            print(f"Training tokenizer on {len(tokens)} bytes. Target merges: {num_merges_target}")

        for merge_idx in range(num_merges_target):
            pair_counts = self._get_pair_counts(tokens)
            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[best_pair] < 1:
                break

            new_id = 256 + merge_idx
            self.merges.append(best_pair)
            self.bpe_ranks[best_pair] = merge_idx
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            # Apply merge across tokens
            new_tokens: List[int] = []
            i = 0
            n = len(tokens)
            p0, p1 = best_pair
            while i < n:
                if i < n - 1 and tokens[i] == p0 and tokens[i + 1] == p1:
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

            if verbose and (merge_idx + 1) % 500 == 0:
                print(f"Merge {merge_idx + 1}/{num_merges_target}: {best_pair} -> {new_id} ({self.vocab[new_id]!r})")

        self.vocab_size = len(self.vocab)
        if verbose:
            print(f"Tokenizer training complete. Final vocab size: {self.vocab_size}")

    def encode(self, text: str) -> List[int]:
        """
        Encode string to list of token IDs using learned BPE merges.
        """
        if not text:
            return []

        tokens = list(text.encode("utf-8"))
        if len(tokens) < 2:
            return tokens

        while len(tokens) >= 2:
            # Find the pair with the lowest merge rank
            best_pair = None
            best_rank = float("inf")

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.bpe_ranks.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            new_id = 256 + self.bpe_ranks[best_pair]
            p0, p1 = best_pair

            new_tokens: List[int] = []
            i = 0
            n = len(tokens)
            while i < n:
                if i < n - 1 and tokens[i] == p0 and tokens[i + 1] == p1:
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def decode(self, ids: List[int]) -> str:
        """
        Decode list of token IDs back into string.
        """
        byte_chunks = [self.vocab.get(i, b"") for i in ids]
        raw_bytes = b"".join(byte_chunks)
        return raw_bytes.decode("utf-8", errors="replace")

    def save(self, path: str):
        """
        Save tokenizer state to JSON file readable by both Python and C.
        """
        data = {
            "vocab_size": self.vocab_size,
            "merges": [list(pair) for pair in self.merges],
            "vocab": {str(k): list(v) for k, v in self.vocab.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        """
        Load tokenizer state from JSON file.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocab_size = data.get("vocab_size", 8000)
        self.merges = [tuple(pair) for pair in data.get("merges", [])]
        self.bpe_ranks = {pair: idx for idx, pair in enumerate(self.merges)}

        self.vocab = {i: bytes([i]) for i in range(256)}
        for idx, (p0, p1) in enumerate(self.merges):
            new_id = 256 + idx
            self.vocab[new_id] = self.vocab[p0] + self.vocab[p1]

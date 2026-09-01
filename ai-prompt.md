# SUB-AI — AI Agent Prompt

> **Read this file fully before writing a single line of code. Delete this file (`ai-prompt.md`) after completing all phases.**

---

## Project Identity

- **Repo:** `subhobhai943/SUB-AI`
- **Brand:** SUB (by Subho)
- **Goal:** Build a language model **entirely from scratch** — custom architecture, custom weight initialization, custom training loop. No Hugging Face weights. No pretrained checkpoints. Every tensor starts as `torch.randn(...)` and is trained on raw text.
- **License:** GPL-2.0

---

## Context: The SUB Ecosystem

This repo is the **training side** of a larger system. The inference side already exists as a C-based SUB LLM engine with:
- Q8_0 quantization (int8 weights + per-row float scales)
- KV-cache lifecycle management
- Streaming token callback (`inference_token_cb_t`)
- Full inference manager (`inference_generate`)
- Hand-rolled tokenizer in C

Your job is to build the Python/PyTorch training stack that produces weights compatible with that C engine.

---

## Rules (Follow Strictly)

1. **No pretrained weights** — do not call `from_pretrained()`, do not download any model checkpoint.
2. **No Hugging Face `transformers` library** — you may use `torch`, `torch.nn`, `torch.nn.functional`, `numpy`, `tiktoken` (for BPE reference only), and standard Python.
3. **All weights initialized from scratch** — use `nn.init.normal_(w, std=0.02)` for linear/embedding weights, `nn.init.zeros_` for biases, `nn.init.ones_` for LayerNorm gain.
4. **All commits go to `main` branch** — no feature branches.
5. **Delete `ai-prompt.md`** as the very last commit after all phases are done.
6. Keep code clean and well-commented. Each file should have a top-of-file docstring explaining what it does.

---

## Architecture Spec

Build a **decoder-only Transformer** (GPT-style) with these default hyperparameters:

```python
# model/config.py
class SUBConfig:
    vocab_size   = 8000      # custom BPE / byte-level
    context_len  = 512       # max sequence length
    n_embd       = 256       # embedding dimension
    n_heads      = 8         # attention heads (head_dim = 32)
    n_layers     = 6         # transformer blocks
    dropout      = 0.1
    bias         = False     # no bias in linear layers (like GPT-NeoX)
    ffn_mult     = 4         # FFN hidden = n_embd * ffn_mult
```

This gives ~10M parameters — trainable on a free Colab T4 GPU.

---

## File-by-File Implementation Plan

### `model/config.py`
- `SUBConfig` dataclass with the hyperparams above
- `small`, `medium`, `large` presets as class methods

### `model/architecture.py`
- `CausalSelfAttention` — multi-head attention with causal mask, `scaled_dot_product_attention`
- `MLP` — two linear layers with GELU activation
- `Block` — LayerNorm → Attention → residual + LayerNorm → MLP → residual
- `SUBModel` — token embedding + positional embedding + N blocks + final LayerNorm + LM head
- `SUBModel.generate(idx, max_new_tokens, temperature, top_k)` method

### `model/init.py`
- `init_weights(model: SUBModel)` — applies the initialization scheme above
- Special scaling for residual projections: `std = 0.02 / sqrt(2 * n_layers)` (GPT-2 style)

### `tokenizer/tokenizer.py`
- Implement a **byte-level BPE tokenizer** from scratch
- `ByteLevelBPETokenizer` class with:
  - `train(corpus: str, vocab_size: int)` — learn merges from raw text
  - `encode(text: str) -> list[int]`
  - `decode(ids: list[int]) -> str`
  - `save(path: str)` / `load(path: str)` — serialize merges + vocab to JSON
- Seed vocab = all 256 byte values (ids 0–255)

### `data/prepare.py`
- Download **TinyShakespeare** as default dataset (`https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`)
- Train the tokenizer on it
- Tokenize the full corpus
- Split 90% train / 10% val
- Save as `data/train.bin` and `data/val.bin` (numpy uint16 arrays, `.npy` format)

### `train.py`
- Load config, model, tokenizer, data
- `DataLoader` — random batch sampler from `.bin` files, returns `(x, y)` tensors on GPU
- Training loop:
  - AdamW optimizer: `lr=6e-4`, `betas=(0.9, 0.95)`, `weight_decay=0.1`
  - Cosine LR scheduler with linear warmup (warmup = 100 steps)
  - Gradient clipping: `max_norm=1.0`
  - Log train loss every 100 steps
  - Eval on val set every 500 steps
  - Save checkpoint to `checkpoints/ckpt_{step}.pt` every 500 steps
  - Resume from latest checkpoint if `--resume` flag passed
- CLI args via `argparse`: `--steps`, `--batch_size`, `--resume`, `--config` (small/medium/large)

### `export.py`
- Load a checkpoint `.pt` file
- Extract all weight tensors
- Write to `model.bin` in the following binary format (compatible with SUB C engine):
  ```
  [header: 256 bytes]
    magic: uint32 = 0x53554241  ('SUBA')
    version: uint32 = 1
    vocab_size, context_len, n_embd, n_heads, n_layers: uint32 x5
    [padding to 256 bytes]
  [weights: all tensors as float32, row-major]
    token_embedding  [vocab_size, n_embd]
    pos_embedding    [context_len, n_embd]
    for each block:
      ln1.weight, ln1.bias
      attn.qkv.weight  [3*n_embd, n_embd]
      attn.proj.weight [n_embd, n_embd]
      ln2.weight, ln2.bias
      mlp.fc1.weight   [4*n_embd, n_embd]
      mlp.fc2.weight   [n_embd, 4*n_embd]
    final_ln.weight, final_ln.bias
    lm_head.weight   [vocab_size, n_embd]
  ```

### `inference.py`
- Quick Python inference test (no C engine needed)
- Load `model.bin` header + weights back into a `SUBModel`
- Run `model.generate()` with a prompt string
- Print generated text to stdout
- Usage: `python inference.py --checkpoint checkpoints/ckpt_5000.pt --prompt "Once upon" --max_tokens 200`

### `requirements.txt`
```
torch>=2.0.0
numpy
requests
```

---

## Commit Convention

Use this format for all commits:
```
[phase] short description
```
Examples:
- `[config] Add SUBConfig dataclass with small/medium/large presets`
- `[arch] Implement CausalSelfAttention and Block`
- `[tokenizer] Byte-level BPE train/encode/decode`
- `[train] AdamW loop with cosine LR and checkpointing`
- `[export] Write model.bin in SUB binary format`
- `[cleanup] Delete ai-prompt.md`

---

## Final Checklist Before Deleting This File

- [ ] `model/config.py` — SUBConfig + presets
- [ ] `model/architecture.py` — full model
- [ ] `model/init.py` — weight init
- [ ] `tokenizer/tokenizer.py` — byte-level BPE
- [ ] `data/prepare.py` — dataset download + tokenize + split
- [ ] `train.py` — full training loop
- [ ] `export.py` — binary weight export
- [ ] `inference.py` — Python inference test
- [ ] `requirements.txt`
- [ ] All files committed to `main`
- [ ] `ai-prompt.md` **deleted** as the last commit

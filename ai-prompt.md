# SUB-AI — AI Agent Prompt

> **Read this file fully before writing a single line of code. Delete this file (`ai-prompt.md`) as your very last commit after all phases are complete.**

---

## Project Identity

- **Repo:** `subhobhai943/SUB-AI`
- **Owner:** Subhadip (brand: SUB)
- **Goal:** Build a language model **entirely from scratch** in Python + PyTorch. Custom architecture, custom weight initialization, custom training loop, custom tokenizer. No Hugging Face weights. No pretrained checkpoints. No `transformers` library. Every tensor starts as `torch.randn(...)` and is learned from raw text.
- **License:** GPL-2.0
- **All commits go to `main` branch only.**

---

## Important Context

This is a greenfield project. There is **no existing inference engine**, no existing model code, and no existing weights. You are building everything from scratch. The only existing files in the repo are `README.md`, `LICENSE`, and this `ai-prompt.md`.

A future goal (Phase 10, out of scope for now) is to build a C inference engine that can load the exported weights — but that does not exist yet and you should NOT reference it as if it does.

---

## Hard Rules

1. **No pretrained weights** — never call `from_pretrained()` or download any model checkpoint.
2. **No Hugging Face `transformers` library** — allowed packages: `torch`, `torch.nn`, `torch.nn.functional`, `numpy`, `requests`, standard Python only.
3. **All weights initialized from scratch:**
   - Linear/embedding weights: `nn.init.normal_(w, mean=0.0, std=0.02)`
   - Biases: `nn.init.zeros_`
   - LayerNorm gain: `nn.init.ones_`, bias: `nn.init.zeros_`
   - Residual projection weights: `std = 0.02 / sqrt(2 * n_layers)` (GPT-2 style depth scaling)
4. **All commits on `main`** — no feature branches.
5. **Delete `ai-prompt.md`** as the very last commit with message `[cleanup] Delete ai-prompt.md`.
6. Each file must have a top-of-file docstring explaining what it does.
7. Keep code clean, readable, and well-commented.

---

## Architecture Spec

Build a **decoder-only Transformer** (GPT-style).

```python
# model/config.py
@dataclass
class SUBConfig:
    vocab_size   : int   = 8000    # learned from byte-level BPE tokenizer
    context_len  : int   = 512     # max sequence length
    n_embd       : int   = 256     # embedding dimension
    n_heads      : int   = 8       # attention heads (head_dim = 32)
    n_layers     : int   = 6       # number of transformer blocks
    dropout      : float = 0.1
    bias         : bool  = False   # no bias in linear layers
    ffn_mult     : int   = 4       # FFN hidden dim = n_embd * ffn_mult
```

This gives ~10M parameters, trainable on a free Colab T4 GPU.

---

## File-by-File Implementation Plan

### `model/config.py`
- `SUBConfig` dataclass with the hyperparams above
- Three class-method presets: `SUBConfig.small()`, `SUBConfig.medium()`, `SUBConfig.large()`
  - small: default above (~10M params)
  - medium: n_embd=512, n_heads=8, n_layers=8 (~50M params)
  - large: n_embd=768, n_heads=12, n_layers=12 (~120M params)

### `model/architecture.py`
- `CausalSelfAttention(config)` — multi-head causal self-attention using `F.scaled_dot_product_attention` with `is_causal=True`
- `MLP(config)` — two `nn.Linear` layers with GELU activation, no bias
- `Block(config)` — Pre-LN: `LayerNorm → Attention → residual` + `LayerNorm → MLP → residual`
- `SUBModel(config)` — token embedding + positional embedding + N blocks + final LayerNorm + LM head (weight-tied to token embedding)
- `SUBModel.forward(idx, targets=None)` — returns `(logits, loss)` where loss is cross-entropy if targets provided
- `SUBModel.generate(idx, max_new_tokens, temperature=1.0, top_k=None)` — autoregressive sampling
- `SUBModel.count_params()` — returns total and non-embedding parameter count

### `model/init.py`
- `init_weights(model: SUBModel)` function
- Iterates over all named modules, applies init rules from the Hard Rules section above
- Residual projections (attn output proj + mlp fc2) use the depth-scaled std

### `tokenizer/tokenizer.py`
- `ByteLevelBPETokenizer` class, fully from scratch (no `tiktoken`, no `sentencepiece`)
- Seed vocabulary = all 256 byte values (ids 0–255)
- Methods:
  - `train(text: str, vocab_size: int)` — learn BPE merges greedily from raw text
  - `encode(text: str) -> list[int]` — apply learned merges
  - `decode(ids: list[int]) -> str` — reconstruct text
  - `save(path: str)` — serialize vocab + merges to JSON
  - `load(path: str)` — deserialize from JSON
  - `__len__()` — returns current vocab size

### `data/prepare.py`
- Downloads **TinyShakespeare** as the default dataset:
  `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`
- Trains the `ByteLevelBPETokenizer` on the full corpus with `vocab_size=8000`
- Saves tokenizer to `data/tokenizer.json`
- Tokenizes the full corpus
- Splits 90% train / 10% val
- Saves as `data/train.npy` and `data/val.npy` (numpy uint16 arrays)
- Prints stats: total tokens, train tokens, val tokens, vocab size

### `train.py`
- Loads config (default: `SUBConfig.small()`), model, tokenizer, and `.npy` data files
- `get_batch(split, config)` — randomly samples `(x, y)` tensors of shape `(batch_size, context_len)` from the data, moved to GPU
- Training loop:
  - **Optimizer:** AdamW, `lr=6e-4`, `betas=(0.9, 0.95)`, `weight_decay=0.1`, no weight decay on bias/LayerNorm
  - **LR Schedule:** cosine decay with linear warmup (warmup_steps=100), min_lr = max_lr / 10
  - **Gradient clipping:** `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
  - Log train loss every 100 steps
  - Evaluate val loss every 500 steps (model in eval mode, no grad)
  - Save checkpoint every 500 steps to `checkpoints/ckpt_{step}.pt` (saves model state, optimizer state, config, step)
- CLI args via `argparse`:
  - `--steps INT` (default 5000)
  - `--batch_size INT` (default 32)
  - `--resume` flag — auto-loads latest checkpoint from `checkpoints/`
  - `--config [small|medium|large]` (default small)

### `export.py`
- Loads a `.pt` checkpoint
- Writes a flat `model.bin` file in the following binary format:
  ```
  Header (256 bytes, little-endian):
    magic     : uint32 = 0x53554241  (ASCII 'SUBA')
    version   : uint32 = 1
    vocab_size: uint32
    ctx_len   : uint32
    n_embd    : uint32
    n_heads   : uint32
    n_layers  : uint32
    [remaining bytes zero-padded to 256 bytes]

  Weights (float32, row-major, no padding between tensors):
    token_embedding    [vocab_size, n_embd]
    pos_embedding      [context_len, n_embd]
    for each block i in 0..n_layers:
      block.ln1.weight   [n_embd]
      block.ln1.bias     [n_embd]
      block.attn.qkv_w   [3*n_embd, n_embd]
      block.attn.proj_w  [n_embd, n_embd]
      block.ln2.weight   [n_embd]
      block.ln2.bias     [n_embd]
      block.mlp.fc1_w    [4*n_embd, n_embd]
      block.mlp.fc2_w    [n_embd, 4*n_embd]
    final_ln.weight    [n_embd]
    final_ln.bias      [n_embd]
    lm_head.weight     [vocab_size, n_embd]  (same data as token_embedding due to weight tying)
  ```
- Usage: `python export.py --checkpoint checkpoints/ckpt_5000.pt --out model.bin`

### `inference.py`
- Loads a checkpoint `.pt` file (not the `.bin` — that's for the future C engine)
- Loads the tokenizer from `data/tokenizer.json`
- Encodes the prompt, runs `model.generate()`, decodes and prints output
- Usage: `python inference.py --checkpoint checkpoints/ckpt_5000.pt --prompt "Once upon a time" --max_tokens 200 --temperature 0.8 --top_k 40`

### `requirements.txt`
```
torch>=2.0.0
numpy
requests
```

---

## Commit Convention

```
[phase] short description
```

Examples:
- `[config] Add SUBConfig dataclass with small/medium/large presets`
- `[arch] Implement CausalSelfAttention, MLP, Block, SUBModel`
- `[init] Custom weight initialization with depth scaling`
- `[tokenizer] Byte-level BPE train/encode/decode/save/load`
- `[data] TinyShakespeare download, tokenize, train/val split`
- `[train] AdamW loop with cosine LR, grad clip, checkpointing`
- `[export] Write model.bin in SUBA binary format`
- `[inference] Python text generation from checkpoint`
- `[cleanup] Delete ai-prompt.md`

---

## Final Checklist (Delete This File Only After All Are Done)

- [ ] `model/config.py` — SUBConfig + small/medium/large presets
- [ ] `model/architecture.py` — full model with generate()
- [ ] `model/init.py` — weight init with depth scaling
- [ ] `tokenizer/tokenizer.py` — byte-level BPE from scratch
- [ ] `data/prepare.py` — download + tokenize + split
- [ ] `train.py` — full training loop with CLI args
- [ ] `export.py` — SUBA binary weight export
- [ ] `inference.py` — prompt → generated text
- [ ] `requirements.txt`
- [ ] All files committed to `main`
- [ ] `ai-prompt.md` **deleted** as the final commit

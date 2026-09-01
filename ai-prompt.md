# SUB-AI — AI Agent Prompt

> **Read this file fully before writing a single line of code. Delete this file (`ai-prompt.md`) as your very last commit after all phases are complete.**

---

## Project Identity

- **Repo:** `subhobhai943/SUB-AI` (public)
- **Owner:** Subhadip (brand: SUB)
- **Goal:** Build a complete AI language model from scratch in one repo — C inference engine first, then the PyTorch training stack. No Hugging Face. No pretrained weights. No `transformers` library.
- **License:** GPL-2.0
- **All commits go to `main` branch only. No feature branches.**

---

## Important Context

This is a **greenfield project**. No model code, no weights, no engine exists yet. The only files currently in the repo are `README.md`, `LICENSE`, and this `ai-prompt.md`.

The build order is intentional:
1. **C engine first** — define the binary weight format before training, so the C loader and the Python exporter always agree
2. **Python architecture** — matches the C engine’s expected tensor layout exactly
3. **Weights from scratch** — randomly initialized, no downloads
4. **Training loop** — PyTorch, runs on Colab T4
5. **Export + integration test** — train in Python, run in C

---

## Hard Rules

1. **No pretrained weights** — never call `from_pretrained()`, never download model checkpoints.
2. **No `transformers` library** — allowed: `torch`, `torch.nn`, `torch.nn.functional`, `numpy`, `requests`, `struct`, `argparse`, standard Python/C only.
3. **Weight init from scratch:**
   - Linear/embedding: `nn.init.normal_(w, mean=0.0, std=0.02)`
   - Bias: `nn.init.zeros_`
   - LayerNorm gain: `nn.init.ones_`, bias: `nn.init.zeros_`
   - Residual projections: `std = 0.02 / sqrt(2 * n_layers)` (GPT-2 depth scaling)
4. **C code:** use only C99 standard library (`stdio.h`, `stdlib.h`, `math.h`, `string.h`, `stdint.h`). No external C libraries.
5. All commits on `main`. Delete this file last.
6. Every file must have a top-of-file comment/docstring explaining what it does.

---

## SUBA Binary Weight Format

This format is the contract between the Python exporter and the C loader. Define it first and never change it without updating both sides.

```
Header (256 bytes, little-endian uint32):
  [0]  magic      = 0x53554241  (ASCII: 'SUBA')
  [1]  version    = 1
  [2]  vocab_size
  [3]  context_len
  [4]  n_embd
  [5]  n_heads
  [6]  n_layers
  [7..63] zero padding

Weights (float32, row-major, packed, no alignment padding):
  token_embedding      [vocab_size, n_embd]
  pos_embedding        [context_len, n_embd]
  for each block i in 0..n_layers-1:
    ln1.weight         [n_embd]
    ln1.bias           [n_embd]
    attn.qkv.weight    [3*n_embd, n_embd]
    attn.proj.weight   [n_embd, n_embd]
    ln2.weight         [n_embd]
    ln2.bias           [n_embd]
    mlp.fc1.weight     [4*n_embd, n_embd]
    mlp.fc2.weight     [n_embd, 4*n_embd]
  final_ln.weight      [n_embd]
  final_ln.bias        [n_embd]
  lm_head.weight       [vocab_size, n_embd]  (weight-tied = same data as token_embedding)
```

---

## Phase 1 — C LLM Engine (`engine/`)

Build each file in this order:

### `engine/loader.h` + `engine/loader.c`
- `ModelHeader` struct matching the binary format above
- `ModelWeights` struct with float pointers to each tensor region
- `load_model(const char *path, ModelHeader *hdr, ModelWeights *w)` — mmap or fread the .bin file, set all pointers into the weight buffer
- `free_model(ModelWeights *w)`

### `engine/matmul.c` + `engine/matmul.h`
- `matmul(float *out, const float *x, const float *w, int n, int d)` — x[n] @ w[d,n]^T → out[d]
- `softmax(float *x, int n)` — in-place numerically stable softmax
- `gelu(float x) -> float` — exact GELU: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))`
- `rmsnorm(float *out, const float *x, const float *w, int n)` — RMS normalization
- `layernorm(float *out, const float *x, const float *w, const float *b, int n)` — standard LayerNorm

### `engine/kvcache.h` + `engine/kvcache.c`
- `KVCache` struct: float arrays for keys and values, shape `[n_layers, context_len, n_heads, head_dim]`
- `kvcache_init(KVCache *kv, ModelHeader *hdr)`
- `kvcache_free(KVCache *kv)`
- `kvcache_reset(KVCache *kv)` — zero the cache for new conversation

### `engine/model.h` + `engine/model.c`
- `transformer_forward(float *logits, int token, int pos, ModelHeader *hdr, ModelWeights *w, KVCache *kv)`
- Implements one forward pass: embed → N blocks (attention + MLP) → final layernorm → lm_head projection
- Attention block: QKV projection → split heads → scaled dot-product with causal mask → write to KV cache → attend → project
- MLP block: fc1 → GELU → fc2

### `engine/sampler.h` + `engine/sampler.c`
- `sample_argmax(float *logits, int vocab_size) -> int`
- `sample_topp(float *logits, int vocab_size, float temperature, float top_p) -> int`
- `sample_topk(float *logits, int vocab_size, float temperature, int top_k) -> int`

### `engine/tokenizer.h` + `engine/tokenizer.c`
- Loads `tokenizer.json` (same file saved by Python tokenizer)
- `tokenizer_encode(Tokenizer *t, const char *text, int *out_ids, int *out_len)`
- `tokenizer_decode(Tokenizer *t, int id) -> const char*`

### `engine/inference.c`
- `main()` — CLI: `./inference --model model.bin --tokenizer data/tokenizer.json --prompt "hello" --max_tokens 200 --temperature 0.8 --top_k 40`
- Loads model + tokenizer, runs autoregressive generation loop, prints tokens as they are generated

### `engine/Makefile`
```makefile
CC = gcc
CFLAGS = -O2 -Wall -std=c99 -lm
SRCS = inference.c loader.c matmul.c kvcache.c model.c sampler.c tokenizer.c
OBJS = $(SRCS:.c=.o)

all: inference

inference: $(OBJS)
	$(CC) $(CFLAGS) -o inference $(OBJS)

clean:
	rm -f $(OBJS) inference
```

---

## Phase 2 — Model Architecture (PyTorch) (`model/`)

### `model/config.py`
```python
@dataclass
class SUBConfig:
    vocab_size  : int   = 8000
    context_len : int   = 512
    n_embd      : int   = 256
    n_heads     : int   = 8       # head_dim = n_embd // n_heads = 32
    n_layers    : int   = 6
    dropout     : float = 0.1
    bias        : bool  = False
    ffn_mult    : int   = 4
```
Add `small()`, `medium()` (n_embd=512, n_layers=8), `large()` (n_embd=768, n_layers=12) class-method presets.

### `model/architecture.py`
- `CausalSelfAttention` — uses `F.scaled_dot_product_attention(is_causal=True)`
- `MLP` — fc1 → GELU → fc2, no bias
- `Block` — Pre-LN residual connections
- `SUBModel` — token emb + pos emb + blocks + final LN + LM head (weight-tied to token emb)
- `SUBModel.forward(idx, targets=None)` returns `(logits, loss)`
- `SUBModel.generate(idx, max_new_tokens, temperature=1.0, top_k=None)`

### `model/init.py`
- `init_weights(model)` applying init rules from Hard Rules above
- Residual proj std = `0.02 / sqrt(2 * n_layers)`

---

## Phase 3 — Tokenizer (Python) (`tokenizer/tokenizer.py`)

- `ByteLevelBPETokenizer` from scratch, no external tokenizer libs
- Seed vocab: 256 byte values (ids 0–255)
- `train(text, vocab_size)`, `encode(text)`, `decode(ids)`, `save(path)`, `load(path)`
- Save format: JSON with `vocab` dict and ordered `merges` list — same file the C tokenizer reads

---

## Phase 4 — Data Pipeline (`data/prepare.py`)

- Download TinyShakespeare from `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`
- Train tokenizer on full corpus, `vocab_size=8000`, save to `data/tokenizer.json`
- Tokenize corpus, split 90/10, save as `data/train.npy` and `data/val.npy` (numpy uint16)
- Print: total tokens, vocab size, train/val split sizes

---

## Phase 5 — Training Loop (`train.py`)

- Load config, build model, apply `init_weights`, move to GPU
- `get_batch(split)` — random (x, y) pairs of shape `(batch_size, context_len)`
- AdamW: `lr=6e-4`, `betas=(0.9,0.95)`, `weight_decay=0.1` (skip decay for 1D params)
- Cosine LR with 100-step linear warmup, min_lr = max_lr / 10
- Grad clip: `clip_grad_norm_(1.0)`
- Log train loss every 100 steps
- Eval val loss every 500 steps
- Save checkpoint every 500 steps: `checkpoints/ckpt_{step}.pt`
- CLI: `--steps`, `--batch_size`, `--resume`, `--config [small|medium|large]`

---

## Phase 6 — Export + Integration

### `export.py`
- Load `.pt` checkpoint, extract tensors in the exact order defined in the SUBA binary format above
- Write header (256 bytes) then all weights as float32 little-endian
- Usage: `python export.py --checkpoint checkpoints/ckpt_5000.pt --out model.bin`

### End-to-end test
- Run `data/prepare.py` → `train.py` → `export.py` → `make -C engine` → `./engine/inference --model model.bin --prompt "Once upon"`
- Confirm tokens stream to stdout from the C engine

---

## Commit Convention

```
[phase] short description
```
- `[engine] Add SUBA loader: ModelHeader, ModelWeights, load_model`
- `[engine] Add matmul, softmax, gelu, layernorm`
- `[engine] Add KV-cache init/reset/free`
- `[engine] Add transformer forward pass`
- `[engine] Add top-k sampler`
- `[engine] Add C tokenizer, Makefile`
- `[arch] Add SUBConfig with presets`
- `[arch] Implement CausalSelfAttention, MLP, Block, SUBModel`
- `[init] Custom weight init with depth scaling`
- `[tokenizer] Byte-level BPE from scratch`
- `[data] TinyShakespeare prep, tokenize, split`
- `[train] AdamW + cosine LR + checkpointing`
- `[export] Write SUBA .bin weight file`
- `[test] End-to-end: Python train → C inference`
- `[cleanup] Delete ai-prompt.md`

---

## Final Checklist

### Phase 1 — C Engine
- [ ] `engine/loader.c` + `engine/loader.h`
- [ ] `engine/matmul.c` + `engine/matmul.h`
- [ ] `engine/kvcache.c` + `engine/kvcache.h`
- [ ] `engine/model.c` + `engine/model.h`
- [ ] `engine/sampler.c` + `engine/sampler.h`
- [ ] `engine/tokenizer.c` + `engine/tokenizer.h`
- [ ] `engine/inference.c`
- [ ] `engine/Makefile`

### Phase 2 — Architecture
- [ ] `model/config.py`
- [ ] `model/architecture.py`
- [ ] `model/init.py`

### Phase 3–6
- [ ] `tokenizer/tokenizer.py`
- [ ] `data/prepare.py`
- [ ] `train.py`
- [ ] `export.py`
- [ ] `requirements.txt`

### Done
- [ ] End-to-end test passes
- [ ] All files committed to `main`
- [ ] `ai-prompt.md` **deleted** as the final commit

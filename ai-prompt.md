# SUB-AI — AI Agent Prompt

> **Read this file fully before writing a single line of code. Delete this file (`ai-prompt.md`) as your very last commit after all phases are complete.**

---

## Project Identity

- **Repo:** `subhobhai943/SUB-AI` (public)
- **Owner:** Subhadip (brand: SUB)
- **Goal:** Build a complete AI language model from scratch in one repo — C inference engine first, then the TensorFlow training stack. No Hugging Face. No pretrained weights. No `transformers` library.
- **Training framework: TensorFlow** (not PyTorch)
- **License:** GPL-2.0
- **All commits go to `main` branch only. No feature branches.**

---

## Important Context

This is a **greenfield project**. No model code, no weights, no engine exists yet. The only files currently in the repo are `README.md`, `LICENSE`, and this `ai-prompt.md`.

The build order is intentional:
1. **C engine first** — define the binary weight format before training, so the C loader and the Python exporter always agree
2. **TensorFlow architecture** — Transformer defined as `tf.keras.layers.Layer` subclasses, matching the C engine’s expected tensor layout exactly
3. **Weights from scratch** — randomly initialized using TensorFlow initializers, no downloads
4. **Training loop** — custom `tf.GradientTape` loop, runs on Colab T4 GPU
5. **Export + integration test** — extract numpy arrays from TF variables, write SUBA `.bin`, run C engine

---

## Hard Rules

1. **No pretrained weights** — never load any model checkpoint from the internet.
2. **No `transformers` library** — allowed packages: `tensorflow`, `numpy`, `requests`, `struct`, `argparse`, standard Python/C only.
3. **Weight init from scratch using TensorFlow initializers:**
   - Embedding + linear kernels: `tf.keras.initializers.TruncatedNormal(stddev=0.02)`
   - Biases: `tf.keras.initializers.Zeros()`
   - LayerNorm gamma: `tf.keras.initializers.Ones()`, beta: `tf.keras.initializers.Zeros()`
   - Residual projection kernels: `stddev = 0.02 / sqrt(2 * n_layers)` (GPT-2 depth scaling)
4. **C code:** C99 standard library only (`stdio.h`, `stdlib.h`, `math.h`, `string.h`, `stdint.h`). No external C libraries.
5. All commits on `main`. Delete this file last.
6. Every file must have a top-of-file comment/docstring explaining what it does.

---

## SUBA Binary Weight Format

This format is the hard contract between the Python exporter and the C loader. Define it first and never change it without updating both sides.

```
Header (256 bytes, little-endian uint32):
  [0]  magic       = 0x53554241  (ASCII: 'SUBA')
  [1]  version     = 1
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
    ln1.gamma          [n_embd]
    ln1.beta           [n_embd]
    attn.qkv_kernel    [n_embd, 3*n_embd]   (TF convention: [in, out])
    attn.proj_kernel   [n_embd, n_embd]
    ln2.gamma          [n_embd]
    ln2.beta           [n_embd]
    mlp.fc1_kernel     [n_embd, 4*n_embd]
    mlp.fc2_kernel     [4*n_embd, n_embd]
  final_ln.gamma       [n_embd]
  final_ln.beta        [n_embd]
  lm_head_kernel       [n_embd, vocab_size]  (weight-tied = transpose of token_embedding)
```

> Note: TensorFlow stores dense kernel weights as `[in_features, out_features]` (column-major convention). The C matmul must account for this — use `x @ W` not `W @ x`.

---

## Phase 1 — C LLM Engine (`engine/`)

### `engine/loader.h` + `engine/loader.c`
- `ModelHeader` struct matching the binary format above
- `ModelWeights` struct with float pointers to each tensor region
- `load_model(const char *path, ModelHeader *hdr, ModelWeights *w)`
- `free_model(ModelWeights *w)`

### `engine/matmul.c` + `engine/matmul.h`
- `matmul(float *out, const float *x, const float *W, int in_dim, int out_dim)` — `x[in] @ W[in, out]` → `out[out]` (TF layout)
- `softmax(float *x, int n)` — in-place numerically stable
- `gelu(float x) -> float` — `0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))`
- `layernorm(float *out, const float *x, const float *gamma, const float *beta, int n)`

### `engine/kvcache.h` + `engine/kvcache.c`
- `KVCache` struct: float arrays shape `[n_layers, context_len, n_heads, head_dim]`
- `kvcache_init`, `kvcache_free`, `kvcache_reset`

### `engine/model.h` + `engine/model.c`
- `transformer_forward(float *logits, int token, int pos, ModelHeader *hdr, ModelWeights *w, KVCache *kv)`
- embed token + pos → N blocks (pre-LN attention + pre-LN MLP) → final LN → lm_head projection
- Causal attention: QKV via matmul → split heads → scaled dot-product with causal mask → write KV cache → attend → project

### `engine/sampler.h` + `engine/sampler.c`
- `sample_argmax(float *logits, int vocab_size) -> int`
- `sample_topk(float *logits, int vocab_size, float temperature, int top_k) -> int`

### `engine/tokenizer.h` + `engine/tokenizer.c`
- Loads `data/tokenizer.json` (same JSON the Python tokenizer saves)
- `tokenizer_encode(Tokenizer *t, const char *text, int *out_ids, int *out_len)`
- `tokenizer_decode_id(Tokenizer *t, int id) -> const char*`

### `engine/inference.c`
- CLI: `./inference --model model.bin --tokenizer data/tokenizer.json --prompt "hello" --max_tokens 200 --temperature 0.8 --top_k 40`
- Autoregressive generation loop, stream tokens to stdout

### `engine/Makefile`
```makefile
CC     = gcc
CFLAGS = -O2 -Wall -std=c99 -lm
SRCS   = inference.c loader.c matmul.c kvcache.c model.c sampler.c tokenizer.c
OBJS   = $(SRCS:.c=.o)

all: inference

inference: $(OBJS)
	$(CC) $(CFLAGS) -o inference $(OBJS)

clean:
	rm -f $(OBJS) inference
```

---

## Phase 2 — Model Architecture (TensorFlow) (`model/`)

### `model/config.py`
```python
from dataclasses import dataclass

@dataclass
class SUBConfig:
    vocab_size  : int   = 8000
    context_len : int   = 512
    n_embd      : int   = 256
    n_heads     : int   = 8
    n_layers    : int   = 6
    dropout     : float = 0.1
    ffn_mult    : int   = 4

    @classmethod
    def small(cls):  return cls()                                          # ~10M
    @classmethod
    def medium(cls): return cls(n_embd=512, n_heads=8,  n_layers=8)       # ~50M
    @classmethod
    def large(cls):  return cls(n_embd=768, n_heads=12, n_layers=12)      # ~120M
```

### `model/architecture.py`
All layers subclass `tf.keras.layers.Layer`:
- `CausalSelfAttention(config)` — QKV as one `Dense(3*n_embd, use_bias=False)`, split, scaled dot-product with causal mask built via `tf.linalg.band_part`, output projection
- `MLP(config)` — `Dense(4*n_embd) → GELU → Dense(n_embd)`, no bias
- `Block(config)` — Pre-LN: `LayerNormalization → Attention → residual` + `LayerNormalization → MLP → residual`
- `SUBModel(config)` — `tf.keras.Model` subclass:
  - token embedding: `tf.keras.layers.Embedding(vocab_size, n_embd)`
  - pos embedding: `tf.keras.layers.Embedding(context_len, n_embd)`
  - N `Block` layers
  - final `LayerNormalization`
  - `call(idx, training=False)` returns logits `[B, T, vocab_size]` (weight-tied lm_head: `logits = x @ tf.transpose(self.token_emb.embeddings)`)
  - `generate(prompt_ids, max_new_tokens, temperature=1.0, top_k=None)` — numpy-based autoregressive loop

### `model/init.py`
- `init_weights(model: SUBModel, config: SUBConfig)` — iterates `model.trainable_variables`, applies init rules from Hard Rules above using `variable.assign()`

---

## Phase 3 — Tokenizer (`tokenizer/tokenizer.py`)

- `ByteLevelBPETokenizer` from scratch, no external tokenizer libs
- Seed vocab: 256 byte values (ids 0–255)
- `train(text, vocab_size)`, `encode(text) -> list[int]`, `decode(ids) -> str`
- `save(path)` / `load(path)` — JSON with `vocab` dict and ordered `merges` list
- Same JSON format the C tokenizer reads

---

## Phase 4 — Data Pipeline (`data/prepare.py`)

- Download TinyShakespeare: `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`
- Train tokenizer, `vocab_size=8000`, save to `data/tokenizer.json`
- Tokenize corpus, 90/10 split, save as `data/train.npy` + `data/val.npy` (numpy uint16)
- Print stats: total tokens, vocab size, split sizes

---

## Phase 5 — Training Loop (`train.py`)

- Load config, build + compile model, call `init_weights`
- `get_batch(data, batch_size, context_len)` — random (x, y) numpy arrays → `tf.constant`
- Custom loop with `tf.GradientTape`:
  ```python
  with tf.GradientTape() as tape:
      logits = model(x, training=True)
      loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
                 y, logits, from_logits=True))
  grads = tape.gradient(loss, model.trainable_variables)
  grads, _ = tf.clip_by_global_norm(grads, 1.0)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))
  ```
- **Optimizer:** `tf.keras.optimizers.Adam(lr_schedule, beta_1=0.9, beta_2=0.95, epsilon=1e-8)`
- **LR schedule:** subclass `tf.keras.optimizers.schedules.LearningRateSchedule`, implement cosine decay with 100-step linear warmup, min_lr = max_lr / 10
- Log train loss every 100 steps
- Eval val loss every 500 steps (call model with `training=False`)
- Save checkpoint every 500 steps using `tf.train.Checkpoint` + `tf.train.CheckpointManager` to `checkpoints/`
- CLI via `argparse`: `--steps`, `--batch_size`, `--resume`, `--config [small|medium|large]`

---

## Phase 6 — Export + Integration

### `export.py`
- Load checkpoint via `tf.train.Checkpoint`
- Extract each weight as `.numpy()` (already float32)
- Write SUBA `.bin` in exact tensor order defined above
- Usage: `python export.py --checkpoint checkpoints/ --out model.bin`

### `requirements.txt`
```
tensorflow>=2.13.0
numpy
requests
```

### End-to-end test
`data/prepare.py` → `train.py` → `export.py` → `make -C engine` → `./engine/inference --model model.bin --prompt "Once upon"`

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
- `[engine] Add C tokenizer + Makefile`
- `[arch] Add SUBConfig with presets`
- `[arch] Implement CausalSelfAttention, MLP, Block, SUBModel in TensorFlow`
- `[init] Custom weight init with depth scaling`
- `[tokenizer] Byte-level BPE from scratch`
- `[data] TinyShakespeare prep, tokenize, split`
- `[train] GradientTape loop + cosine LR + TF checkpointing`
- `[export] Write SUBA .bin weight file from TF variables`
- `[test] End-to-end: TF train → C inference`
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

### Phase 2 — Architecture (TensorFlow)
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
- [ ] End-to-end test passes (TF train → C inference)
- [ ] All files committed to `main`
- [ ] `ai-prompt.md` **deleted** as the final commit

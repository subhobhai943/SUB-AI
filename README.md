# SUB-AI

> A language model built entirely from scratch — C inference engine, custom architecture, custom weights, trained from zero. No pretrained models. No Hugging Face.

![License](https://img.shields.io/badge/license-GPL--2.0-blue.svg)
![Status](https://img.shields.io/badge/status-in%20development-yellow)
![Language](https://img.shields.io/badge/language-C%20%7C%20Python%20%7C%20PyTorch-orange)

---

## What is SUB-AI?

SUB-AI is a fully hand-crafted AI language model built under the **SUB** brand by [Subhadip](https://subhobhai943.xyz).

Everything is written from scratch in this single repository — the C inference engine, the model architecture, the weights, and the training pipeline:

- **C LLM Engine** — low-level inference runtime: tokenizer, matrix ops, KV-cache, sampling (no external libs)
- **Architecture** — custom decoder-only Transformer defined in Python + PyTorch (no `transformers` library)
- **Weights** — randomly initialized, trained from zero on raw text data
- **Tokenizer** — byte-level BPE implemented from scratch in both C and Python
- **Training loop** — AdamW + cosine LR scheduler written manually in PyTorch
- **Weight export** — trained PyTorch weights exported to a flat binary format the C engine loads natively

---

## Build Order

The project is built in this deliberate sequence:

1. **C LLM Engine** — build the inference runtime in C first so the binary format is locked before training
2. **Model Architecture** — define the Transformer in PyTorch matching the C engine’s expected layout
3. **Weight Initialization** — proper from-scratch init schemes (Kaiming, depth scaling)
4. **Tokenizer** — byte-level BPE in both Python (for training) and C (for inference)
5. **Training Loop** — AdamW + cosine LR, checkpoint save/resume, runs on Colab T4 GPU
6. **Export** — write trained weights to binary format the C engine reads directly
7. **End-to-end test** — train in Python, run inference in C

---

## Project Structure

```
SUB-AI/
├── engine/                     # C inference engine
│   ├── tokenizer.c / .h        # Byte-level BPE tokenizer in C
│   ├── matmul.c / .h           # Matrix multiply, softmax, GELU ops
│   ├── kvcache.c / .h          # Key-value cache for autoregressive inference
│   ├── model.c / .h            # Transformer forward pass in C
│   ├── sampler.c / .h          # Temperature + top-k sampling
│   ├── loader.c / .h           # Load .bin weight file into memory
│   ├── inference.c             # Main inference entry point (CLI)
│   └── Makefile
├── model/                      # PyTorch model (training side)
│   ├── architecture.py         # Transformer blocks, attention, MLP, full model
│   ├── config.py               # SUBConfig dataclass + presets
│   └── init.py                 # Custom weight initialization
├── tokenizer/
│   └── tokenizer.py            # Byte-level BPE (Python, for training)
├── data/
│   └── prepare.py              # Dataset download, tokenize, train/val split
├── train.py                    # Main PyTorch training loop
├── export.py                   # Export weights → SUBA .bin format for C engine
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Roadmap

### Phase 1 — C LLM Engine
- [ ] 1.1 — Define SUBA binary weight format (`engine/loader.h`)
- [ ] 1.2 — Matrix ops: matmul, softmax, GELU, RMSNorm (`engine/matmul.c`)
- [ ] 1.3 — KV-cache implementation (`engine/kvcache.c`)
- [ ] 1.4 — Transformer forward pass in C (`engine/model.c`)
- [ ] 1.5 — Temperature + top-k sampler (`engine/sampler.c`)
- [ ] 1.6 — Byte-level BPE tokenizer in C (`engine/tokenizer.c`)
- [ ] 1.7 — Weight file loader (`engine/loader.c`)
- [ ] 1.8 — CLI inference entry point (`engine/inference.c`)
- [ ] 1.9 — Makefile + build test

### Phase 2 — Model Architecture (PyTorch)
- [ ] 2.1 — `SUBConfig` dataclass with small/medium/large presets
- [ ] 2.2 — `CausalSelfAttention`, `MLP`, `Block`, `SUBModel`
- [ ] 2.3 — Custom weight initialization (`model/init.py`)

### Phase 3 — Tokenizer (Python)
- [ ] 3.1 — Byte-level BPE: train, encode, decode, save, load

### Phase 4 — Data Pipeline
- [ ] 4.1 — Download TinyShakespeare, train tokenizer, save train/val splits

### Phase 5 — Training Loop
- [ ] 5.1 — AdamW + cosine LR + grad clipping
- [ ] 5.2 — Checkpoint save and resume
- [ ] 5.3 — Val loss eval every N steps

### Phase 6 — Export + Integration
- [ ] 6.1 — Export PyTorch weights to SUBA `.bin` format
- [ ] 6.2 — End-to-end test: train in Python, run inference in C

---

## Default Hyperparameters (~10M params)

| Parameter | Value |
|---|---|
| Vocabulary size | 8,000 |
| Context length | 512 tokens |
| Embedding dimension | 256 |
| Attention heads | 8 (head dim = 32) |
| Transformer layers | 6 |
| FFN multiplier | 4× |
| Dropout | 0.1 |
| Optimizer | AdamW |
| Learning rate | 6e-4 (cosine decay) |
| Target hardware | Google Colab T4 GPU |

---

## License

This project is licensed under the **GNU General Public License v2.0**. See [LICENSE](./LICENSE) for details.

---

> Part of the **SUB** open-source ecosystem — [subhobhai943.xyz](https://subhobhai943.xyz)

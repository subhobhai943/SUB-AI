# SUB-AI

> A from-scratch AI language model — custom architecture, custom weights, no pretrained models.

![License](https://img.shields.io/badge/license-GPL--2.0-blue.svg)
![Status](https://img.shields.io/badge/status-in%20development-yellow)
![Language](https://img.shields.io/badge/language-Python%20%7C%20C-lightgrey)

---

## What is SUB-AI?

SUB-AI is a fully hand-crafted language model built under the **SUB** brand. The entire architecture — embeddings, attention blocks, feedforward layers, weight initialization, and training loop — is written from scratch. No Hugging Face weights. No pretrained checkpoints. Every tensor starts as a random number and is trained on raw text data.

This is the training-side counterpart to the **SUB LLM inference engine** (C-based), which already implements Q8_0 quantization, a KV-cache, a streaming token callback, and a full inference manager.

---

## Goals

- Design a custom Transformer-based architecture with original hyperparameters
- Initialize all weights from scratch (Kaiming / custom schemes)
- Train on a curated text corpus using PyTorch
- Export weights to a binary format compatible with the SUB C inference engine
- Run quantized inference natively inside the SUB OS ecosystem

---

## Project Structure

```
SUB-AI/
├── model/
│   ├── architecture.py   # Transformer blocks, attention, MLP
│   ├── config.py         # Hyperparameter configs
│   └── init.py           # Custom weight initialization
├── tokenizer/
│   └── tokenizer.py      # Byte-level or custom BPE tokenizer
├── data/
│   └── prepare.py        # Dataset downloading and preprocessing
├── train.py              # Main training loop (PyTorch)
├── export.py             # Export weights → SUB binary format
├── inference.py          # Quick inference test (pure Python)
├── requirements.txt
└── README.md
```

---

## Roadmap

- [ ] Phase 1 — Architecture design (`model/architecture.py`)
- [ ] Phase 2 — Custom weight initialization (`model/init.py`)
- [ ] Phase 3 — Byte-level tokenizer (`tokenizer/tokenizer.py`)
- [ ] Phase 4 — Dataset prep script (`data/prepare.py`)
- [ ] Phase 5 — Training loop with AdamW + LR scheduler (`train.py`)
- [ ] Phase 6 — Checkpoint saving / resuming
- [ ] Phase 7 — Weight export to SUB binary format (`export.py`)
- [ ] Phase 8 — Integration test with SUB C inference engine
- [ ] Phase 9 — Scale up: larger model, more data

---

## Tech Stack

| Layer | Tool |
|---|---|
| Architecture & Training | Python + PyTorch |
| Inference (production) | C (SUB engine) |
| Quantization | Q8_0 (int8 + per-row scale) |
| Target hardware | Colab T4 GPU / CUDA |

---

## License

This project is licensed under the **GNU General Public License v2.0**. See [LICENSE](./LICENSE) for details.

---

> Part of the **SUB** open-source ecosystem — [subhobhai943.xyz](https://subhobhai943.xyz)

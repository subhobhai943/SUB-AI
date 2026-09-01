# SUB-AI

> A language model built entirely from scratch — custom architecture, custom weights, trained from zero. No pretrained models. No Hugging Face.

![License](https://img.shields.io/badge/license-GPL--2.0-blue.svg)
![Status](https://img.shields.io/badge/status-in%20development-yellow)
![Language](https://img.shields.io/badge/language-Python%20%7C%20PyTorch-orange)

---

## What is SUB-AI?

SUB-AI is a fully hand-crafted language model built under the **SUB** brand by [Subhadip](https://subhobhai943.xyz). Every component is written from scratch:

- **Architecture** — custom decoder-only Transformer (no `transformers` library)
- **Weights** — randomly initialized, trained from zero on raw text
- **Tokenizer** — byte-level BPE implemented from scratch
- **Training loop** — AdamW + cosine LR scheduler, written manually
- **Inference** — pure Python sampling loop (no external inference engines)

This is Phase 1 of a larger vision: once the model trains successfully, a future **SUB C inference engine** will run the exported weights natively — but that comes later.

---

## Goals

- Design a custom Transformer architecture with original hyperparameters
- Initialize all weights from scratch using proper init schemes
- Train on a curated text corpus using PyTorch on Google Colab (T4 GPU)
- Save and resume training checkpoints
- Export trained weights to a flat binary format for future C engine integration
- Demonstrate that a real LLM can be built without any pretrained model

---

## Project Structure

```
SUB-AI/
├── model/
│   ├── architecture.py   # Transformer blocks, attention, MLP, full model
│   ├── config.py         # SUBConfig dataclass with hyperparameter presets
│   └── init.py           # Custom weight initialization scheme
├── tokenizer/
│   └── tokenizer.py      # Byte-level BPE tokenizer (train/encode/decode/save/load)
├── data/
│   └── prepare.py        # Dataset download, tokenize, train/val split
├── train.py              # Main training loop (AdamW, cosine LR, checkpointing)
├── export.py             # Export trained weights to SUB binary format (.bin)
├── inference.py          # Python inference: load checkpoint, generate text
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Roadmap

- [ ] Phase 1 — Architecture design (`model/architecture.py`)
- [ ] Phase 2 — Custom weight initialization (`model/init.py`)
- [ ] Phase 3 — Byte-level BPE tokenizer (`tokenizer/tokenizer.py`)
- [ ] Phase 4 — Dataset prep: download + tokenize + split (`data/prepare.py`)
- [ ] Phase 5 — Training loop with AdamW + cosine LR scheduler (`train.py`)
- [ ] Phase 6 — Checkpoint saving and resuming
- [ ] Phase 7 — Weight export to binary format (`export.py`)
- [ ] Phase 8 — Python inference test (`inference.py`)
- [ ] Phase 9 — Scale up: larger model, bigger dataset
- [ ] Phase 10 — *(Future)* Build SUB C inference engine to run exported weights natively

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

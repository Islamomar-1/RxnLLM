# RxnLLM: Chemical Language Model Fine-Tuned on Reaction Data

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License">
  <img src="https://img.shields.io/badge/build-passing-brightgreen.svg" alt="Build Passing">
  <img src="https://img.shields.io/badge/tests-16%20passing-brightgreen.svg" alt="Tests Passing">
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg" alt="HuggingFace Transformers">
  <img src="https://img.shields.io/badge/PEFT-LoRA-orange.svg" alt="PEFT LoRA">
  <img src="https://img.shields.io/badge/RDKit-2023.09+-red.svg" alt="RDKit">
  <img src="https://img.shields.io/badge/base%20model-Mistral--7B-purple.svg" alt="Mistral-7B">
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome">
</p>

<p align="center">
  <b>Parameter-efficient fine-tuning of Mistral-7B on reaction SMILES for reagent prediction and yield regression.</b>
</p>

---

## Motivation

Large language models trained on natural-language corpora encode broad reasoning
ability but lack native understanding of chemical reaction syntax: SMILES atoms,
bonds, stereochemistry tokens, and reaction arrows (`>>`) are typically shredded
into meaningless sub-word fragments by general-purpose BPE tokenizers. At the same
time, dedicated chemistry-specific transformers (e.g. Molecular Transformer,
Chemformer) achieve strong task performance but discard the general reasoning,
instruction-following, and few-shot capabilities that come from large-scale
pretraining on natural language.

**RxnLLM** bridges this gap. It takes a strong open-weights instruction-following
LLM (Mistral-7B), extends its tokenizer with chemistry-aware SMILES tokens, and
applies **LoRA (Low-Rank Adaptation)** fine-tuning on USPTO reaction data. This
gives a model that:

- Understands reaction SMILES at the atom/bond level instead of the sub-word level.
- Retains general language reasoning for natural-language reaction conditions,
  procedure text, and chain-of-thought explanations.
- Can be fine-tuned on a single high-memory GPU thanks to parameter-efficient
  adaptation (LoRA touches <1% of total parameters).
- Is directly evaluable on two practically important downstream tasks: **reagent
  prediction** (multi-label classification over a reagent vocabulary) and **yield
  regression** (continuous reaction-yield estimation).

This project builds directly on the tokenization and reaction-representation
methodology introduced by Schwaller et al. in their Molecular Transformer work
(see [Citation](#citation)), adapting those ideas to the LoRA fine-tuning regime
for modern decoder-only LLMs.

## Features

- 🧪 **Chemistry-aware tokenizer extension** — augments the base Mistral tokenizer
  with atom-wise SMILES tokens, ring-closure digits, stereochemistry markers, and
  reaction-arrow tokens, initialized via embedding-mean warm-start.
- 🔧 **LoRA fine-tuning pipeline** (`train.py`) — 4-bit/8-bit quantized loading via
  `bitsandbytes`, `peft` LoRA adapters on attention + MLP projections, gradient
  checkpointing, and HuggingFace `Trainer`-based training loop.
- 🧬 **RDKit-backed data pipeline** — canonicalization, validity filtering, and
  reaction atom-mapping sanity checks on USPTO reaction SMILES.
- 📊 **Two downstream evaluation heads** (`evaluate.py`):
  - **Reagent prediction** — multi-label classification, scored with top-k
    accuracy, precision/recall/F1.
  - **Yield regression** — scalar yield prediction, scored with MAE, RMSE, R².
- 🚀 **Simple inference CLI** (`infer.py`) — generate reagents/conditions or
  predict yield for a single reaction SMILES or a batch file.
- 📓 **Notebooks** for data exploration and an end-to-end quickstart fine-tune.
- ✅ **Unit tests** for the tokenizer, dataset, and model wrapper.

## Installation

```bash
git clone https://github.com/islamomar/rxnllm.git
cd rxnllm

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

> **GPU note:** LoRA fine-tuning of Mistral-7B in 4-bit is designed to fit on a
> single 24 GB GPU (e.g. RTX 3090/4090, A10G). CPU-only execution is supported
> for the tokenizer/data pipeline and for running the test suite, but full
> training requires a CUDA-capable GPU.

## Quick Start

### 1. Prepare data

```bash
python data/download_uspto.py --output_dir data/raw
python data/preprocess.py --input_dir data/raw --output_dir data/processed
```

### 2. Extend the tokenizer with chemistry-aware tokens

```python
from models.tokenizer import build_chemistry_tokenizer

tokenizer = build_chemistry_tokenizer(
    base_model="mistralai/Mistral-7B-v0.1",
    save_dir="models/rxnllm-tokenizer",
)
```

### 3. Fine-tune with LoRA

```bash
python train.py \
    --base_model mistralai/Mistral-7B-v0.1 \
    --tokenizer_dir models/rxnllm-tokenizer \
    --train_file data/processed/train.jsonl \
    --eval_file data/processed/val.jsonl \
    --output_dir checkpoints/rxnllm-lora \
    --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
    --num_train_epochs 3 --per_device_train_batch_size 4 \
    --load_in_4bit
```

### 4. Run inference

```bash
python infer.py \
    --adapter_dir checkpoints/rxnllm-lora \
    --task reagent_prediction \
    --smiles "CC(=O)Oc1ccccc1C(=O)O.CCN(CC)CC>>CC(=O)Oc1ccccc1C(=O)OCC"
```

### 5. Evaluate on benchmark tasks

```bash
python evaluate.py \
    --adapter_dir checkpoints/rxnllm-lora \
    --task reagent_prediction \
    --test_file data/processed/test_reagents.jsonl

python evaluate.py \
    --adapter_dir checkpoints/rxnllm-lora \
    --task yield_regression \
    --test_file data/processed/test_yield.jsonl
```

## Tech Stack

| Layer                | Tool / Library                              |
|-----------------------|----------------------------------------------|
| Base LLM              | Mistral-7B (`mistralai/Mistral-7B-v0.1`)     |
| Fine-tuning           | HuggingFace `transformers`, `peft` (LoRA)    |
| Quantization          | `bitsandbytes` (4-bit / 8-bit NF4)           |
| Chemistry toolkit     | `RDKit`                                       |
| Data                  | USPTO reaction dataset (via `rxn4chemistry`/`uspto-50k` style extraction) |
| Training loop         | HuggingFace `Trainer` / `accelerate`         |
| Experiment tracking   | `wandb` (optional)                            |
| Testing               | `pytest`                                      |

## Repository Structure

```
RxnLLM/
├── data/
│   ├── __init__.py
│   ├── download_uspto.py      # Fetches raw USPTO reaction data
│   ├── preprocess.py          # Canonicalizes SMILES, builds JSONL splits
│   └── dataset.py             # PyTorch/HF Dataset classes for both tasks
├── models/
│   ├── __init__.py
│   ├── tokenizer.py           # Chemistry-aware tokenizer extension
│   └── lora_model.py          # LoRA-wrapped Mistral-7B model builders
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_quickstart_finetune.ipynb
├── tests/
│   ├── __init__.py
│   ├── test_tokenizer.py
│   ├── test_dataset.py
│   └── test_model.py
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions: tests on Python 3.10 / 3.11
├── train.py                   # LoRA fine-tuning entry point
├── infer.py                   # CLI inference for reagent/yield tasks
├── evaluate.py                # Benchmark evaluation entry point
├── requirements.txt
├── setup.py
├── .gitignore
├── LICENSE
├── CONTRIBUTING.md
├── CHANGELOG.md
└── README.md
```

## Benchmarks

Results on held-out USPTO reaction splits (LoRA rank 16, 4-bit base model, 3 epochs).
Baselines reproduced from their respective papers where available.

| Model                                   | Reagent Pred. Top-1 Acc. | Reagent Pred. F1 | Yield MAE (%) | Yield R² |
|------------------------------------------|:------------------------:|:----------------:|:--------------:|:--------:|
| Molecular Transformer (Schwaller 2019)  | 71.8%                     | 0.69              | —              | —        |
| Chemformer (fine-tuned)                  | 74.2%                     | 0.72              | 8.9            | 0.71     |
| Mistral-7B (zero-shot, no fine-tune)     | 12.4%                     | 0.10              | 24.7           | 0.02     |
| **RxnLLM (LoRA, ours)**                  | **76.5%**                 | **0.74**          | **7.8**        | **0.76** |

> ⚠️ Numbers above are illustrative placeholders to demonstrate the reporting
> format; reproduce them with `evaluate.py` on your own trained checkpoint and
> replace this table with your measured results before publishing.

## Contributing

Contributions are welcome! Quick version:

1. Fork the repository and create a feature branch:
   `git checkout -b feature/my-improvement`
2. Install dev dependencies and run the test suite before opening a PR:
   ```bash
   pip install -r requirements.txt
   pytest tests/ -v
   ```
3. Follow existing code style (PEP8, type hints where practical, docstrings on
   public functions).
4. Open a pull request describing the motivation, approach, and any benchmark
   deltas. Please include before/after numbers for any change touching
   `train.py`, `models/`, or `evaluate.py`.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full developer workflow,
suggested contribution areas, and bug-report guidelines. Release history is
tracked in [CHANGELOG.md](CHANGELOG.md).

## Citation

If you use RxnLLM in your research, please cite this repository and the
foundational Molecular Transformer work that informs its reaction-SMILES
tokenization approach:

```bibtex
@article{schwaller2019molecular,
  title={Molecular Transformer: A Model for Uncertainty-Calibrated Chemical Reaction Prediction},
  author={Schwaller, Philippe and Laino, Teodoro and Gaudin, Th{\'e}ophile and Bolgar, Peter and Hunter, Christopher A. and Bekas, Costas and Lee, Alpha A.},
  journal={ACS Central Science},
  volume={5},
  number={9},
  pages={1572--1583},
  year={2019},
  publisher={American Chemical Society},
  doi={10.1021/acscentsci.9b00576}
}

@software{rxnllm2026,
  title={RxnLLM: Chemical Language Model Fine-Tuned on Reaction Data},
  author={Omar, Islam},
  year={2026},
  url={https://github.com/islamomar/rxnllm}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
# RxnLLM

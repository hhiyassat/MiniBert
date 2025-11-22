# Mini BERT - Masked Language Model

A lightweight BERT implementation for Arabic text training with two different training workflows.

---

## 📋 Table of Contents

1. [train.py Workflow](#trainpy-workflow) - Standard training with WordPiece tokenizer
2. [train_bpe.py Workflow](#train_bpepy-workflow) - Training with BPE tokenizer
3. [Shared Components](#shared-components)
4. [Requirements](#requirements)

---

## 🚀 train.py Workflow

Standard training pipeline using WordPiece tokenization for Arabic text.

### Quick Start

```bash
# Training from scratch
python train.py

# Resume training from a checkpoint
python train.py model/mini_bert_mlm.pt

# Resume from a specific model folder
python train.py model/

# Inference
python inference.py
```

### Files Structure

```
train.py workflow:
├── train.py              # Main training script
├── config.py             # Configuration parameters (shared)
├── model.py              # MiniBERT architecture (shared)
├── dataset.py            # MLM dataset processing (shared)
├── tokenizer_utils.py    # WordPiece tokenizer training
├── trainer.py            # Training loop functions (shared)
├── utils.py              # Text preprocessing utilities
├── inference.py          # Inference script for trained models
└── model/                # Saved models directory
    ├── mini_bert_mlm.pt  # Best trained model
    ├── tokenizer.json    # Trained tokenizer
    └── vocab.txt         # Vocabulary list
```

### Key Files

| File | Description |
|------|-------------|
| `train.py` | Main training entry point. Handles dataset loading, tokenizer training, model initialization, and training orchestration |
| `tokenizer_utils.py` | Trains WordPiece tokenizer from Arabic text corpus |
| `utils.py` | Arabic text preprocessing functions (cleaning, normalization) |
| `inference.py` | Loads trained model and performs masked language model predictions |

### Features

- **WordPiece Tokenization**: Uses WordPiece algorithm (same as BERT)
- **Arabic Text Support**: Preprocesses and tokenizes Arabic text
- **Checkpoint Management**: Automatic saving of best models and resumable training
- **Validation Split**: Optional validation set for monitoring overfitting
- **EMA Loss Tracking**: Exponential moving average for smoother loss visualization

### Training Configuration

Edit `config.py` to customize:
- Vocabulary size
- Model architecture (hidden dim, layers, heads)
- Training hyperparameters (learning rate, batch size, epochs)
- Validation split ratio
- Checkpoint frequency

### Example Usage

```bash
# Start fresh training
python train.py

# Resume from checkpoint
python train.py model/mini_bert_mlm.pt

# Run inference on trained model
python inference.py
```

---

## 🔤 train_bpe.py Workflow

Training pipeline using Byte-Pair Encoding (BPE) tokenization, optimized for Arabic text with syllable-aware tokenization.

### Quick Start

```bash
# Training with BPE tokenizer
python train_bpe.py

# Inference with BPE model
python inference_bpe.py

# Inference with 100 examples
python inference_bpe_100.py
```

### Files Structure

```
train_bpe.py workflow:
├── train_bpe.py          # Main BPE training script
├── config.py             # Configuration parameters (shared)
├── model.py              # MiniBERT architecture (shared)
├── dataset.py            # MLM dataset processing (shared)
├── trainer.py            # Training loop functions (shared)
├── inference_bpe.py      # BPE inference script
├── inference_bpe_100.py  # BPE inference with 100 examples
└── model-bpe/            # Saved BPE models directory
    ├── mini_bert_mlm.pt  # Best trained model
    ├── tokenizer.json    # BPE tokenizer
    └── vocab.txt         # Vocabulary list
```

### Key Files

| File | Description |
|------|-------------|
| `train_bpe.py` | Main BPE training script. Handles BPE tokenizer creation, Arabic text processing, and model training |
| `inference_bpe.py` | Inference script for BPE-trained models with Arabic text handling |
| `inference_bpe_100.py` | Extended inference script that processes 100 examples with detailed output |

### Features

- **BPE Tokenization**: Byte-Pair Encoding for subword tokenization
- **Arabic-Specific Processing**: Handles Arabic consonants, diacritics, and syllable-aware tokenization
- **Quran Text Support**: Optimized for processing Quranic text
- **Advanced Inference**: Multiple inference scripts with different output formats

### Training Configuration

Uses the same `config.py` as train.py workflow. The BPE tokenizer is trained separately within `train_bpe.py`.

### Example Usage

```bash
# Train with BPE tokenizer
python train_bpe.py

# Run inference
python inference_bpe.py

# Run inference on 100 examples
python inference_bpe_100.py
```

---

## 🔗 Shared Components

These files are used by both workflows:

| File | Description |
|------|-------------|
| `config.py` | Central configuration file with model and training hyperparameters |
| `model.py` | MiniBERT model architecture (Transformer encoder with MLM head) |
| `dataset.py` | MLMDataset class for creating masked language modeling datasets |
| `trainer.py` | Training loop, checkpoint management, and loss tracking functions |

### Model Architecture

The MiniBERT model consists of:
- **8 Transformer Encoder Layers** - Learn language patterns
- **Token Embeddings** - Convert tokens to 512-dimensional vectors
- **Position Embeddings** - Add sequence position information
- **MLM Head** - Predict masked tokens

### Model Specifications

| Parameter | Value |
|-----------|-------|
| Vocabulary size | 63,584 tokens (configurable) |
| Hidden dimension | 512 |
| Encoder layers | 8 |
| Attention heads | 8 |
| Feed-forward dimension | 2,048 |
| Max sequence length | 256 tokens |
| Total parameters | ~90.5 million |

---

## 📊 Training Features

### Checkpoint Management

Both workflows support automatic checkpoint saving:

**Checkpoint Types:**
- **Best model**: `mini_bert_mlm.pt` - Saved when loss improves
- **Periodic checkpoints**: Saved every N batches (configurable via `check_every_batches`)

**Resume Training:**
```bash
# Continue from checkpoint
python train.py model/mini_bert_mlm.pt

# Or resume from model folder
python train.py model/
```

When resuming:
- ✅ Model weights are restored
- ✅ Optimizer and scheduler states are restored
- ✅ Training continues from the next epoch
- ✅ Best loss tracking is preserved
- ✅ Global step counter is maintained

### Loss Tracking

- **Real-time EMA Loss**: Exponential moving average for smooth loss visualization
- **Best Loss Tracking**: Tracks best training loss (for train.py) or validation loss (when validation enabled)
- **Progress Bar**: Shows current loss, EMA loss, best loss, learning rate, and step count

### MLM Training Task

**Masked Language Modeling:**
1. Randomly mask 15% of tokens in text (configurable via `mlm_prob`)
2. Model learns to predict masked tokens
3. Example: "The [MASK] is running" → predicts "dog"

---

## 🎯 Trained Models

This repository contains several pre-trained models trained on different dataset sizes. The folder names indicate the training data size (e.g., `750K` means 750,000 training samples).

### Available Models

| Model Folder | Training Data Size | Best Loss | Epochs | Global Steps | Description |
|--------------|-------------------|-----------|--------|--------------|-------------|
| `model/` | 1,000,000 samples | **5.7423** | 15 | 928,125 | Default model folder - trained on 1M Arabic samples |
| `model 1000000 ver1_edition/` | 1,000,000 samples | **5.7423** | 15 | 928,125 | Alternative 1M sample model |
| `model_750K_ver1_edition/` | 750,000 samples | **5.2692** | 10 | 232,040 | First version trained on 750K samples |
| `model_750K_ver2_edition/` | 750,000 samples | **5.2692** | 10 | 232,040 | Second version trained on 750K samples |
| `model_500000/` | 500,000 samples | **8.2942** | 1 | - | Early training checkpoint on 500K samples |
| `model_eng/` | English dataset | **3.5428** | 20 | - | Model trained on English WikiText-2 dataset |

### Model Performance Notes

- **Lower loss = better performance**: The best loss represents the lowest validation/training loss achieved during training
- **750K models** show the best performance (loss: 5.27) among Arabic models
- **1M models** have slightly higher loss (5.74) but were trained for more epochs (15 vs 10)
- **English model** has the lowest loss (3.54) as English is generally easier to model than Arabic
- **500K model** is an early checkpoint with higher loss, indicating it needs more training

### Using Pre-trained Models

To use any of these models for inference or to resume training:

```bash
# Use a specific model folder
python train.py model_750K_ver1_edition/

# Or specify the checkpoint file directly
python train.py model_750K_ver1_edition/mini_bert_mlm.pt

# For inference
python inference.py  # Uses model/ by default, or modify the path in the script
```

Each model folder contains:
- `mini_bert_mlm.pt` - The trained model checkpoint with best loss
- `tokenizer.json` - The tokenizer used for training
- `vocab.txt` - Vocabulary file (if available)

---

## 📁 Directory Structure

```
training_bert/
├── train.py                  # Standard training script
├── train_bpe.py              # BPE training script
├── config.py                 # Shared configuration
├── model.py                  # Shared model architecture
├── dataset.py                # Shared dataset class
├── trainer.py                # Shared training functions
├── tokenizer_utils.py        # WordPiece tokenizer (train.py)
├── utils.py                  # Text preprocessing (train.py)
├── inference.py              # Standard inference (train.py)
├── inference_bpe.py          # BPE inference (train_bpe.py)
├── inference_bpe_100.py      # BPE inference 100 examples (train_bpe.py)
│
├── model/                    # Default output (1M samples, loss: 5.74)
│   ├── mini_bert_mlm.pt
│   ├── tokenizer.json
│   └── vocab.txt
│
├── model 1000000 ver1_edition/  # 1M samples model (loss: 5.74)
│   ├── mini_bert_mlm.pt
│   ├── tokenizer.json
│   └── vocab.txt
│
├── model_750K_ver1_edition/  # 750K samples v1 (loss: 5.27)
│   ├── mini_bert_mlm.pt
│   └── tokenizer.json
│
├── model_750K_ver2_edition/  # 750K samples v2 (loss: 5.27)
│   ├── mini_bert_mlm.pt
│   ├── tokenizer.json
│   └── vocab.txt
│
├── model_500000/             # 500K samples (loss: 8.29, early checkpoint)
│   ├── mini_bert_mlm.pt
│   ├── tokenizer.json
│   └── vocab.txt
│
├── model_eng/                # English dataset model (loss: 3.54)
│   ├── mini_bert_mlm.pt
│   ├── mini_bert_mlm_epoch*.pt  # Multiple epoch checkpoints
│   ├── tokenizer.json
│   └── vocab.txt
│
├── model-bpe/                # train_bpe.py output directory
│   ├── mini_bert_mlm.pt
│   ├── tokenizer.json
│   └── vocab.txt
│
└── samples/                  # Sample text outputs
    ├── sample.txt
    └── sample_syllabified.txt
```

---

## 🔧 Requirements

- Python 3.8+
- PyTorch
- Tokenizers (Hugging Face)
- Transformers (Hugging Face)
- NumPy
- TQDM

Install dependencies:
```bash
pip install torch tokenizers transformers numpy tqdm
```

---

## 📝 Notes

- Both workflows can be used independently
- Models trained with one tokenizer cannot be used with the other's inference script
- Checkpoint files are compatible between workflows if using the same model architecture
- Training data paths are configurable in the respective training scripts
# MiniBert

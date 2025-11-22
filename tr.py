import torch
import numpy as np
from torch.utils.data import DataLoader
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from config import CONFIG
from model import MiniBERT
from dataset import MLMDataset
from tokenizer_utils import train_tokenizer
from trainer import train_mlm, get_linear_schedule_with_warmup
from utils import preprocess_arabic_text


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Check if resuming from checkpoint
    resume_from = None
    if len(sys.argv) > 1:
        resume_from = sys.argv[1]
        print(f"Resuming training from checkpoint: {resume_from}\n")
    
    print("=" * 70)
    print("LOADING ARABIC DATASET")
    print("=" * 70)
    
    texts = []
    corpus_path = "/home/ubuntu/PycharmProjects/Qwen_aric/bert-cv/raw_data/arabic_super_cleaned.txt"
#Tokenizer dir: /home/ubuntu/PycharmProjects/Qwen_aric/bert-cv/raw_data/arabic_syllable_tokenizer_clean
    print(f"Loading full dataset from: {corpus_path}")
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 500000:
                break
            text = line.strip()
            if len(text) > 0:
                text = preprocess_arabic_text(text)
                if len(text) > 50:
                    texts.append(text)
    
    if CONFIG['train_size'] is not None:
        texts = texts[:CONFIG['train_size']]
    
    print(f"✓ Dataset loaded from: {corpus_path}")
    print(f"✓ Dataset size: {len(texts):,} texts\n")
    
    print("=" * 70)
    print("TRAINING TOKENIZER")
    print("=" * 70)
    tokenizer = train_tokenizer(texts, vocab_size=CONFIG['vocab_size'])
    vocab_size = tokenizer.get_vocab_size()
    print()
    
    print("=" * 70)
    print("CREATING DATALOADER")
    print("=" * 70)
    mlm_dataset = MLMDataset(texts, tokenizer, max_len=CONFIG['max_len'], mlm_prob=CONFIG['mlm_prob'])
    train_loader = DataLoader(
        mlm_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    print(f"✓ DataLoader created")
    print(f"  Batch size: {CONFIG['batch_size']}")
    print(f"  Total batches per epoch: {len(train_loader):,}\n")
    
    print("=" * 70)
    print("INITIALIZING MODEL")
    print("=" * 70)
    
    # If resuming, get vocab_size from checkpoint
    if resume_from is not None:
        print(f"Reading vocab_size from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        checkpoint_vocab_size = checkpoint['model_state_dict']['embedding.token_embedding.weight'].shape[0]
        print(f"✓ Checkpoint vocab_size: {checkpoint_vocab_size}")
        vocab_size = checkpoint_vocab_size
    
    model = MiniBERT(
        vocab_size=vocab_size,
        hidden_dim=CONFIG['hidden_dim'],
        num_layers=CONFIG['num_layers'],
        num_heads=CONFIG['num_heads'],
        intermediate_dim=CONFIG['intermediate_dim'],
        max_len=CONFIG['max_len'],
        dropout=CONFIG['dropout']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model initialized")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Hidden dim: {CONFIG['hidden_dim']}")
    print(f"  Layers: {CONFIG['num_layers']}")
    print(f"  Heads: {CONFIG['num_heads']}\n")
    
    print("=" * 70)
    print("SETUP OPTIMIZER & SCHEDULER")
    print("=" * 70)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=CONFIG['weight_decay']
    )
    
    total_steps = len(train_loader) * CONFIG['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=CONFIG['warmup_steps'],
        num_training_steps=total_steps
    )
    
    print(f"✓ Optimizer: AdamW (lr={CONFIG['learning_rate']})")
    print(f"✓ Scheduler: Linear warmup ({CONFIG['warmup_steps']} steps) + decay")
    print(f"  Total training steps: {total_steps:,}\n")
    
    print("=" * 70)
    print("TRAINING MODEL")
    print("=" * 70)
    print(f"Training for {CONFIG['epochs']} epochs...\n")
    os.makedirs("model", exist_ok=True)
    
    tokenizer.save("model/tokenizer.json")
    model = train_mlm(
        model,
        train_loader,
        optimizer,
        scheduler,
        device,
        epochs=CONFIG['epochs'],
        config=CONFIG,
        save_path="model/mini_bert_mlm.pt",
        resume_from=resume_from
    )
    
    print("=" * 70)
    print("SAVING FINAL MODEL")
    print("=" * 70)
    
    # Save sample texts
    os.makedirs("samples", exist_ok=True)
    num_samples = min(100, len(texts))  # Save up to 100 samples
    with open("samples/sample.txt", "w", encoding="utf-8") as f:
        for text in texts[:num_samples]:
            f.write(text + "\n")
    print(f"✓ Sample texts saved to: samples/sample.txt ({num_samples} samples)")
    
    vocab = tokenizer.get_vocab()
    with open("model/vocab.txt", "w", encoding="utf-8") as f:
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
        for token, token_id in sorted_vocab:
            f.write(f"{token}\n")
    
    print("✓ Model saved to: model/mini_bert_mlm.pt")
    print("✓ Tokenizer saved to: model/tokenizer.json")
    print("✓ Vocabulary saved to: model/vocab.txt")
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED!")
    print("=" * 70)
    print(f"\nConfiguration used:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

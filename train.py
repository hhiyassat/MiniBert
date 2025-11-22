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
    save_folder = "model"  # Default save folder
    
    if len(sys.argv) > 1:
        provided_path = sys.argv[1]
        
        # Determine if path is a folder or file
        if os.path.isdir(provided_path):
            # Path is a folder - check for model and tokenizer inside
            save_folder = provided_path.rstrip('/')
            model_path = os.path.join(save_folder, "mini_bert_mlm.pt")
            tokenizer_path_in_folder = os.path.join(save_folder, "tokenizer.json")
            
            if os.path.exists(model_path) and os.path.exists(tokenizer_path_in_folder):
                resume_from = model_path
                print(f"✓ Found model checkpoint: {model_path}")
                print(f"✓ Found tokenizer: {tokenizer_path_in_folder}")
                print(f"Resuming training from folder: {save_folder}\n")
            else:
                missing = []
                if not os.path.exists(model_path):
                    missing.append(f"Model checkpoint ({model_path})")
                if not os.path.exists(tokenizer_path_in_folder):
                    missing.append(f"Tokenizer ({tokenizer_path_in_folder})")
                
                print("=" * 70)
                print("ERROR: Cannot resume training")
                print("=" * 70)
                print(f"Folder provided: {save_folder}")
                print(f"Missing files: {', '.join(missing)}")
                print("\nPlease run 'python train.py' first to create the model and tokenizer.")
                print("=" * 70)
                sys.exit(1)
        elif os.path.isfile(provided_path):
            # Path is a file (direct checkpoint path)
            resume_from = provided_path
            save_folder = os.path.dirname(resume_from) or "model"
            
            # Check if tokenizer exists in same folder
            tokenizer_path_in_folder = os.path.join(save_folder, "tokenizer.json")
            if not os.path.exists(tokenizer_path_in_folder):
                print("=" * 70)
                print("ERROR: Cannot resume training")
                print("=" * 70)
                print(f"Model checkpoint found: {resume_from}")
                print(f"Tokenizer missing: {tokenizer_path_in_folder}")
                print("\nPlease make sure both model and tokenizer exist in the same folder.")
                print("Or run 'python train.py' first to create them.")
                print("=" * 70)
                sys.exit(1)
            
            print(f"✓ Found model checkpoint: {resume_from}")
            print(f"✓ Found tokenizer: {tokenizer_path_in_folder}")
            print(f"Resuming training from: {resume_from}\n")
        else:
            # Path doesn't exist
            print("=" * 70)
            print("ERROR: Path does not exist")
            print("=" * 70)
            print(f"Provided path: {provided_path}")
            print("\nPlease provide a valid folder path (e.g., 'model/') or checkpoint file.")
            print("Or run 'python train.py' without arguments to start fresh training.")
            print("=" * 70)
            sys.exit(1)
    else:
        print("Starting fresh training (no checkpoint provided)\n")
    
    print("=" * 70)
    print("LOADING ARABIC DATASET")
    print("=" * 70)
    
    texts = []
    corpus_path = "/home/ubuntu/PycharmProjects/Qwen_aric/bert-cv/raw_data/arabic_super_cleaned.txt"
#Tokenizer dir: /home/ubuntu/PycharmProjects/Qwen_aric/bert-cv/raw_data/arabic_syllable_tokenizer_clean
    print(f"Loading full dataset from: {corpus_path}")
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            text = line.strip()
            if len(text) > 0:
                text = preprocess_arabic_text(text)
                if len(text) > 0:
                    texts.append(text)
    
    # Use 750k samples for this training
    train_size = 1000000
    if train_size is not None and len(texts) > train_size:
        texts = texts[:train_size]
    
    print(f"✓ Dataset loaded from: {corpus_path}")
    print(f"✓ Dataset size: {len(texts):,} texts\n")
    
    # Split dataset into train and validation
    val_split = CONFIG.get('val_split', 0.01)
    if val_split > 0 and val_split < 1:
        split_idx = int(len(texts) * (1 - val_split))
        train_texts = texts[:split_idx]
        val_texts = texts[split_idx:]
        print(f"✓ Dataset split:")
        print(f"  Training: {len(train_texts):,} texts ({100*(1-val_split):.1f}%)")
        print(f"  Validation: {len(val_texts):,} texts ({100*val_split:.1f}%)\n")
    else:
        train_texts = texts
        val_texts = []
        print("✓ Using full dataset for training (validation disabled)\n")
    
    print("=" * 70)
    print("LOADING/TRAINING TOKENIZER")
    print("=" * 70)
    
    # Check if tokenizer already exists (when resuming)
    tokenizer_path = os.path.join(save_folder, "tokenizer.json")
    if os.path.exists(tokenizer_path) and resume_from is not None:
        print(f"Loading existing tokenizer from: {tokenizer_path}")
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(tokenizer_path)
        print(f"✓ Tokenizer loaded with vocabulary size: {tokenizer.get_vocab_size()}")
    else:
        print("Training new tokenizer...")
        # Use training texts only for tokenizer training
        tokenizer = train_tokenizer(train_texts, vocab_size=CONFIG['vocab_size'])
        print(f"✓ Tokenizer trained with vocabulary size: {tokenizer.get_vocab_size()}")
    
    vocab_size = tokenizer.get_vocab_size()
    print()
    
    print("=" * 70)
    print("CREATING DATALOADER")
    print("=" * 70)
    train_dataset = MLMDataset(train_texts, tokenizer, max_len=CONFIG['max_len'], mlm_prob=CONFIG['mlm_prob'])
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    print(f"✓ Training DataLoader created")
    print(f"  Batch size: {CONFIG['batch_size']}")
    print(f"  Total batches per epoch: {len(train_loader):,}")
    
    # Create validation dataloader if validation split is enabled
    val_loader = None
    if len(val_texts) > 0:
        val_dataset = MLMDataset(val_texts, tokenizer, max_len=CONFIG['max_len'], mlm_prob=CONFIG['mlm_prob'])
        val_loader = DataLoader(
            val_dataset,
            batch_size=CONFIG['batch_size'],
            shuffle=False,  # No need to shuffle validation data
            num_workers=0,
            pin_memory=True if device.type == 'cuda' else False
        )
        print(f"✓ Validation DataLoader created")
        print(f"  Validation batches: {len(val_loader):,}\n")
    else:
        print()
    
    print("=" * 70)
    print("INITIALIZING MODEL")
    print("=" * 70)
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
    
    # Save folder is already determined above
    if resume_from is not None:
        print(f"Resuming training - saving to: {save_folder}/\n")
    else:
        print(f"Starting fresh training - saving to: {save_folder}/\n")
    
    os.makedirs(save_folder, exist_ok=True)
    save_path = f"{save_folder}/mini_bert_mlm.pt"
    tokenizer_path = f"{save_folder}/tokenizer.json"
    
    tokenizer.save(tokenizer_path)
    model = train_mlm(
        model,
        train_loader,
        optimizer,
        scheduler,
        device,
        epochs=CONFIG['epochs'],
        config=CONFIG,
        save_path=save_path,
        resume_from=resume_from,
        val_loader=val_loader
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
    vocab_file = f"{save_folder}/vocab.txt"
    with open(vocab_file, "w", encoding="utf-8") as f:
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
        for token, token_id in sorted_vocab:
            f.write(f"{token}\n")
    
    print(f"✓ Model saved to: {save_folder}/mini_bert_mlm.pt")
    print(f"✓ Tokenizer saved to: {save_folder}/tokenizer.json")
    print(f"✓ Vocabulary saved to: {save_folder}/vocab.txt")
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED!")
    print("=" * 70)
    print(f"\nConfiguration used:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

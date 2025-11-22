CONFIG = {
        'vocab_size': 63584,
        'hidden_dim': 512,
        'num_layers': 8,
        'num_heads': 8,
        'intermediate_dim': 2048,
        'max_len': 256,
        'dropout': 0.1,
        'batch_size': 16,  # Reduced from 32 to avoid CUDA out of memory
        'learning_rate': 1e-4,  # Reduced from 3e-4 for more stable BERT pretraining
        'epochs': 30,
        'warmup_steps': 11700,  # 1% of total steps (1.17M total steps ≈ 11.7k warmup). For 10% warmup, use 117000
        'weight_decay': 0.01,
        'max_grad_norm': 1.0,
        'train_size': None,
        'mlm_prob': 0.15,
        'check_every_batches': 100,  # Changed to 100 as requested
        'patience': 5,
        'ema_decay': 0.99,  # Exponential moving average decay for smoother loss tracking
        'val_split': 0.01,  # 1% of data for validation (set to 0 to disable validation)
    }
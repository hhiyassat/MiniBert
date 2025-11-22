import torch
from torch.utils.data import Dataset
import numpy as np


class MLMDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=128, mlm_prob=0.15):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mlm_prob = mlm_prob
        self.mask_token_id = tokenizer.token_to_id("[MASK]")
        self.pad_token_id = tokenizer.token_to_id("[PAD]")
        self.cls_token_id = tokenizer.token_to_id("[CLS]")
        self.sep_token_id = tokenizer.token_to_id("[SEP]")
        unk_token_id = tokenizer.token_to_id("[UNK]")
        self.vocab_size = tokenizer.get_vocab_size()
        
        # All special tokens that should not be used for random replacement
        self.special_tokens = {
            self.mask_token_id, self.pad_token_id,
            self.cls_token_id, self.sep_token_id
        }
        if unk_token_id is not None:
            self.special_tokens.add(unk_token_id)
        
        # Create list of valid token IDs for random replacement (exclude special tokens)
        self.valid_token_ids = [i for i in range(self.vocab_size) if i not in self.special_tokens]
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        encoding = self.tokenizer.encode(text)
        tokens = encoding.ids
        
        if len(tokens) > self.max_len - 2:
            tokens = tokens[:self.max_len - 2]
        
        tokens = [self.cls_token_id] + tokens + [self.sep_token_id]
        attention_mask = [1] * len(tokens)
        
        padding_len = self.max_len - len(tokens)
        tokens = tokens + [self.pad_token_id] * padding_len
        attention_mask = attention_mask + [0] * padding_len
        
        labels = tokens.copy()
        tokens, labels = self.mask_tokens(tokens, labels)
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
    
    def mask_tokens(self, tokens, labels):
        masked_tokens = tokens.copy()
        
        for i in range(len(tokens)):
            if tokens[i] in self.special_tokens:
                labels[i] = -100
                continue
            
            if np.random.random() < self.mlm_prob:
                prob = np.random.random()
                
                if prob < 0.8:
                    # 80%: Replace with [MASK] token
                    masked_tokens[i] = self.mask_token_id
                elif prob < 0.9:
                    # 10%: Replace with random token from valid vocabulary (excluding special tokens)
                    if len(self.valid_token_ids) > 0:
                        masked_tokens[i] = np.random.choice(self.valid_token_ids)
                    else:
                        # Fallback: if no valid tokens, use mask token
                        masked_tokens[i] = self.mask_token_id
                # 10%: Keep original token unchanged (do nothing)
            else:
                # Not selected for masking: ignore in loss calculation
                labels[i] = -100
        
        return masked_tokens, labels

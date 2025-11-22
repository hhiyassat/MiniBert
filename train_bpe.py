#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import re
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import List, Tuple
from collections import OrderedDict

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from config import CONFIG
from model import MiniBERT
from dataset import MLMDataset
from trainer import train_mlm, get_linear_schedule_with_warmup
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

BASE_DIR = "/home/ubuntu/PycharmProjects/Qwen_aric/training_bert/"
CORPUS_PATH = os.path.join(BASE_DIR, "quran-simple-processed.txt")
TOKENIZER_FILE = "tokenizer.json"
MODEL_OUT_DIR = "model-bpe"
SAMPLES_DIR = "samples"
MODEL_PATH = os.path.join(MODEL_OUT_DIR, "mini_bert_mlm.pt")

TRUE_CONSONANTS = set("ءبتثجحخدذرزسشصضطظعغفقكلمنهة")
AMBIGUOUS_LETTERS = {'و', 'ي', 'ى'}
PURE_LONG_VOWELS = {'ا'}
SHORT_VOWELS = {'َ', 'ُ', 'ِ'}
SUKUN = 'ْ'
SHADDA = 'ّ'
TANWEEN = {'ً', 'ٌ', 'ٍ'}
ALIFS = {'ا', 'أ', 'إ', 'آ', 'ٱ'}
DIACS = SHORT_VOWELS | {SUKUN, SHADDA} | TANWEEN

def normalize_word(word: str) -> str:
    word = re.sub(r'ـ+', '', word)
    word = word.replace('آ', 'ءَ')
    word = word.replace('ؤُ', 'ءُ')
    word = word.replace('ؤْ', 'ءْ')
    word = word.replace('ئِ', 'ءِ')
    word = word.replace('ئْ', 'ءْ')
    word = re.sub(r'أَ([أإآ])', r'ءَ\1', word)
    word = re.sub(r'^أَ', 'ءَ', word)
    word = re.sub(r'أَ$', 'ءَ', word)
    return word

def starts_with_al(word: str) -> bool:
    if not word: return False
    if word[0] in ALIFS:
        i = 1
        while i < len(word) and word[i] in DIACS: i += 1
        return i < len(word) and word[i] == 'ل'
    return False

def normalize_initial_al(word: str) -> str:
    if not starts_with_al(word): return word
    i = 1
    while i < len(word) and word[i] in DIACS: i += 1
    j = i + 1
    while j < len(word) and word[j] in DIACS: j += 1
    return "ءَلْ" + word[j:]

def expand_shadda_both_orders(word: str) -> str:
    out, i, n = [], 0, len(word)
    while i < n:
        ch = word[i]
        if (ch in TRUE_CONSONANTS | AMBIGUOUS_LETTERS) and i + 2 < n:
            if word[i+1] == SHADDA and word[i+2] in (SHORT_VOWELS | TANWEEN):
                out.extend([ch, SUKUN, ch, word[i+2]]); i += 3; continue
            if word[i+1] in (SHORT_VOWELS | TANWEEN) and word[i+2] == SHADDA:
                out.extend([ch, SUKUN, ch, word[i+1]]); i += 3; continue
        out.append(ch); i += 1
    return ''.join(out)

def add_implicit_sukuns(word: str) -> str:
    res, i, n = [], 0, len(word)
    while i < n:
        ch = word[i]
        if ch in DIACS: res.append(ch); i += 1; continue
        if ch in PURE_LONG_VOWELS: res.append(ch); i += 1; continue
        if ch in TRUE_CONSONANTS | AMBIGUOUS_LETTERS:
            res.append(ch)
            if i + 1 < n and word[i+1] in (SHORT_VOWELS | TANWEEN | {SUKUN}):
                i += 1; continue
            if ch in AMBIGUOUS_LETTERS and len(res) >= 2 and res[-2] in SHORT_VOWELS:
                prev_v = res[-2]
                if (prev_v == 'ُ' and ch == 'و') or (prev_v == 'ِ' and ch in {'ي', 'ى'}):
                    if i + 1 < n and word[i+1] in SHORT_VOWELS:
                        res.append(SUKUN); i += 1; continue
                    i += 1; continue
            res.append(SUKUN); i += 1; continue
        res.append(ch); i += 1
    return ''.join(res)

def collapse_double_long_vowels_for_syllabification(src: str) -> str:
    out, i, n = [], 0, len(src)
    while i < n:
        ch = src[i]; out.append(ch)
        if ch in {'و', 'ي', 'ى'} and i + 1 < n and src[i+1] != SUKUN \
           and i + 1 < n and src[i+1] in PURE_LONG_VOWELS:
            i += 2; continue
        i += 1
    return ''.join(out)

def can_start_syll(src: str, i: int) -> bool:
    if i >= len(src): return False
    ch = src[i]
    if i == 0 and ch in PURE_LONG_VOWELS: return True
    if ch not in (TRUE_CONSONANTS | AMBIGUOUS_LETTERS): return False
    if i + 1 < len(src) and src[i+1] == SUKUN: return False
    return True

def take_syll(src: str, i: int) -> Tuple[str, int]:
    n = len(src)
    if not can_start_syll(src, i): return "", i
    if i == 0 and src[i] in PURE_LONG_VOWELS:
        j, coda = i + 1, 0
        while coda < 2 and j + 1 < n and src[j] in (TRUE_CONSONANTS | AMBIGUOUS_LETTERS) \
              and src[j+1] == SUKUN:
            j += 2; coda += 1
        return src[i:j], j
    j = i + 1
    if j >= n or src[j] not in (SHORT_VOWELS | TANWEEN): return "", i
    j += 1
    if j < n and src[j] in (PURE_LONG_VOWELS | AMBIGUOUS_LETTERS):
        if not (src[j] in AMBIGUOUS_LETTERS and j + 1 < n and src[j+1] in SHORT_VOWELS):
            j += 1
    coda = 0
    while coda < 2 and j + 1 < n and src[j] in (TRUE_CONSONANTS | AMBIGUOUS_LETTERS) \
          and src[j+1] == SUKUN:
        j += 2; coda += 1
    return src[i:j], j

def build_parsing_source(word: str) -> str:
    w = normalize_word(word)
    w = normalize_initial_al(w)
    w = expand_shadda_both_orders(w)
    if len(w) >= 3 and w[0] in (TRUE_CONSONANTS | AMBIGUOUS_LETTERS) \
       and w[1] == SUKUN and w[2] == w[0]:
        w = w[2:]
    w = add_implicit_sukuns(w)
    w = collapse_double_long_vowels_for_syllabification(w)
    return w

def syllabify_word(word: str) -> List[str]:
    src = build_parsing_source(word)
    syls, i = [], 0
    while i < len(src):
        syl, j = take_syll(src, i)
        if not syl: return []
        syls.append(syl); i = j
    return syls if ''.join(syls) == src else []

def syllabify_text(line: str) -> str:
    out = []
    for w in line.split():
        s = syllabify_word(w)
        if s:
            s_no_sukun = [syl.replace(SUKUN, '') for syl in s if syl.replace(SUKUN, '')]
            out.extend(s_no_sukun)
    return " ".join(out)

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resume_from = None
    if len(sys.argv) > 1:
        resume_from = sys.argv[1]
    if not os.path.exists(CORPUS_PATH):
        raise SystemExit(f"Corpus not found: {CORPUS_PATH}")
    texts = []
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 100000:
                break
            t = line.strip()
            if not t:
                continue
            if len(t) > 0:
                texts.append(t)
    if CONFIG.get('train_size') is not None:
        texts = texts[:CONFIG['train_size']]
    texts = [syllabify_text(t) for t in texts]
    texts = [t for t in texts if len(t.split()) >= 1]
    if len(texts) == 0:
        raise SystemExit("ERROR: No texts left after syllabification.")
    if not os.path.exists(TOKENIZER_FILE):
        raise SystemExit(f"Tokenizer file not found: {TOKENIZER_FILE}")
    tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
    tokenizer_hf = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    vocab_size = tokenizer.get_vocab_size()
    mlm_dataset = MLMDataset(
        texts,
        tokenizer,
        max_len=CONFIG['max_len'],
        mlm_prob=CONFIG['mlm_prob']
    )
    train_loader = DataLoader(
        mlm_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == 'cuda')
    )
    model = MiniBERT(
        vocab_size=vocab_size,
        hidden_dim=CONFIG['hidden_dim'],
        num_layers=CONFIG['num_layers'],
        num_heads=CONFIG['num_heads'],
        intermediate_dim=CONFIG['intermediate_dim'],
        max_len=CONFIG['max_len'],
        dropout=CONFIG['dropout']
    ).to(device)
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
    os.makedirs(MODEL_OUT_DIR, exist_ok=True)
    tokenizer_hf.save_pretrained(MODEL_OUT_DIR)
    
    import torch.nn.functional as F
    from tqdm import tqdm
    
    best_loss = float('inf')
    global_step = 0
    start_epoch = 0
    check_every_batches = CONFIG['check_every_batches']
    batch_losses = []
    
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
        if start_epoch >= CONFIG['epochs']:
            return
    
    for epoch in range(start_epoch, CONFIG['epochs']):
        model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG['max_grad_norm'])
            optimizer.step()
            scheduler.step()
            
            loss_value = loss.item()
            total_loss += loss_value
            num_batches += 1
            global_step += 1
            batch_losses.append(loss_value)
            
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f'{loss_value:.4f}',
                'best': f'{best_loss:.4f}' if best_loss != float('inf') else 'inf',
                'lr': f'{current_lr:.2e}',
                'step': global_step
            })
            
            if global_step % check_every_batches == 0:
                recent_losses = [l for l in batch_losses[-check_every_batches:] if not (np.isnan(l) or np.isinf(l))]
                if len(recent_losses) > 0:
                    avg_loss_batch = sum(recent_losses) / len(recent_losses)
                    
                    if avg_loss_batch < best_loss:
                        best_loss = avg_loss_batch
                        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'loss': best_loss,
                            'config': CONFIG
                        }, MODEL_PATH)
        
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{CONFIG['epochs']} - Avg Loss: {avg_loss:.4f}, Best: {best_loss:.4f}")
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    num_samples = min(100, len(texts))
    with open(os.path.join(SAMPLES_DIR, "sample_syllabified.txt"), "w", encoding="utf-8") as f:
        for text in texts[:num_samples]:
            f.write(text + "\n")
    vocab = tokenizer.get_vocab()
    with open(os.path.join(MODEL_OUT_DIR, "vocab.txt"), "w", encoding="utf-8") as f:
        for token, token_id in sorted(vocab.items(), key=lambda x: x[1]):
            f.write(f"{token}\n")

if __name__ == "__main__":
    main()


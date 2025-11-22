#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# MiniBERT MLM Inference with Arabic Syllable Tokenizer
# - Loads model trained with syllable tokenizer (tokenizer.wordlevel.json)
# - Syllabifies input text before tokenization
# - Predicts masked tokens syllable-by-syllable
# - 100 EXAMPLES FROM QURAN TEXT
#
# Usage:
#   python inference_bpe_100.py
# =============================================================================

import torch
import torch.nn as nn
from tokenizers import Tokenizer
import numpy as np
import random

from model import MiniBERT
from config import CONFIG
# NOTE: Do NOT import preprocess_arabic_text - it removes diacritics we need for syllabification


# =============================================================================
# Syllabifier (same as train_bpe.py)
# =============================================================================
_CONSONANTS = set('ءأإآابتثجحخدذرزسشصضطظعغفقكلمنهوىيؤئىةٱ')
_SHORT_VOWELS = {'َ', 'ُ', 'ِ'}
_LONG_VOWELS = {'ا', 'و', 'ي', 'ى'}
_SUKUN = 'ْ'
_SHADDA = 'ّ'
_TANWEEN = {'ً', 'ٌ', 'ٍ'}

def syllabify_word(word: str):
    """Convert a word into legal Arabic syllables."""
    ch = list(word)
    i = 0
    syls = []

    while i < len(ch):
        if ch[i] not in _CONSONANTS:
            i += 1
            continue

        syl = ch[i]
        i += 1

        if i < len(ch) and ch[i] == _SHADDA:
            syl += ch[i]
            i += 1
            if i >= len(ch) or ch[i] not in _SHORT_VOWELS:
                syls.append(syl)  # Keep syllable even if short
                continue
            syl += ch[i]
            i += 1
        else:
            if i >= len(ch) or not (ch[i] in _SHORT_VOWELS or ch[i] in _TANWEEN):
                syls.append(syl)  # Keep syllable even if short
                continue
            syl += ch[i]
            i += 1

        if i < len(ch) and ch[i] in _LONG_VOWELS:
            syl += ch[i]
            i += 1
            for _ in range(2):
                if i + 1 < len(ch) and (ch[i] in _CONSONANTS) and (ch[i+1] == _SUKUN):
                    syl += ch[i] + ch[i+1]
                    i += 2
                else:
                    break
        else:
            for _ in range(2):
                if i + 1 < len(ch) and (ch[i] in _CONSONANTS) and (ch[i+1] == _SUKUN):
                    syl += ch[i] + ch[i+1]
                    i += 2
                else:
                    break

        syls.append(syl)  # Keep ALL syllables (including single-char ones)

    return syls

def syllabify_text(line: str) -> str:
    """Turn a sentence into whitespace-separated syllables."""
    out = []
    for w in line.split():
        if w in ['[MASK]', '[CLS]', '[SEP]', '[PAD]', '[UNK]']:
            out.append(w)
        else:
            s = syllabify_word(w)
            if s:
                out.extend(s)
    return " ".join(out)


# =============================================================================
# Model loading and inference
# =============================================================================

def load_model_and_tokenizer(model_path="model-bpe/mini_bert_mlm.pt", 
                            tokenizer_path="syllable_tokenizer.json", 
                            device="cpu"):
    """Load the syllable-based model and tokenizer."""
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    print(f"✓ Tokenizer loaded from {tokenizer_path}")
    
    print("Loading model...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Use CONFIG values if available
    config = checkpoint.get('config', CONFIG)
    state_dict = checkpoint['model_state_dict']
    vocab_size = config.get('vocab_size', tokenizer.get_vocab_size())
    
    model = MiniBERT(
        vocab_size=vocab_size,
        hidden_dim=config.get('hidden_dim', 512),
        num_layers=config.get('num_layers', 8),
        num_heads=config.get('num_heads', 8),
        intermediate_dim=config.get('intermediate_dim', 2048),
        max_len=config.get('max_len', 256),
        dropout=config.get('dropout', 0.1)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded from {model_path}")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Hidden dimension: {config.get('hidden_dim', 512)}")
    print(f"  Number of layers: {config.get('num_layers', 8)}")
    if 'epoch' in checkpoint:
        print(f"  Training epoch: {checkpoint['epoch'] + 1}")
    if 'loss' in checkpoint:
        print(f"  Training loss: {checkpoint['loss']:.4f}")
    print(f"  Device: {device}\n")
    
    return model, tokenizer


def predict_word_iteratively(model, tokenizer, context_left, context_right, device, 
                             max_syllables=10, top_k=5, stop_threshold=0.1, verbose=False):
    """Predict a complete word syllable-by-syllable."""
    model.eval()
    
    # Do NOT preprocess - it removes diacritics needed for syllabification
    # Just syllabify the context as-is
    context_left = syllabify_text(context_left) if context_left else ""
    context_right = syllabify_text(context_right) if context_right else ""
    
    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")
    mask_id = tokenizer.token_to_id("[MASK]")
    
    candidates = [([],  0.0)]
    final_candidates = []
    
    for step in range(max_syllables):
        new_candidates = []
        
        for syllables, log_prob in candidates:
            left_tokens = tokenizer.encode(context_left).ids if context_left else []
            right_tokens = tokenizer.encode(context_right).ids if context_right else []
            
            current_word_tokens = []
            if syllables:
                current_word_syllabified = " ".join(syllables)
                current_word_tokens = tokenizer.encode(current_word_syllabified).ids
            
            tokens = [cls_id] + left_tokens + current_word_tokens + [mask_id] + right_tokens + [sep_id]
            
            input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
            
            with torch.no_grad():
                logits = model(input_ids)
                probabilities = torch.softmax(logits, dim=-1)
            
            mask_pos = tokens.index(mask_id)
            top_k_probs, top_k_indices = torch.topk(probabilities[0, mask_pos], top_k * 2)
            
            for prob, token_id in zip(top_k_probs, top_k_indices):
                next_syllable = tokenizer.id_to_token(token_id.item())
                
                if next_syllable in ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]']:
                    continue
                
                if prob.item() < stop_threshold and len(syllables) > 0:
                    # Store this as a final candidate (stopped early)
                    final_candidates.append((
                        syllables.copy(),
                        log_prob,
                        ''.join(syllables)
                    ))
                    continue
                
                new_syllables = syllables + [next_syllable]
                new_log_prob = log_prob + torch.log(prob).item()
                new_candidates.append((new_syllables, new_log_prob))
                
                if len(new_candidates) >= top_k:
                    break
            
            if len(new_candidates) >= top_k:
                break
        
        if not new_candidates:
            break
        
        new_candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = new_candidates[:top_k]
    
    for syllables, log_prob in candidates:
        if syllables:
            final_candidates.append((
                syllables.copy(),
                log_prob,
                ''.join(syllables)
            ))
    
    seen = set()
    unique_candidates = []
    for syllables, log_prob, word in final_candidates:
        if word not in seen:
            seen.add(word)
            unique_candidates.append({
                'word': word,
                'syllables': syllables,
                'log_probability': log_prob,
                'probability': np.exp(log_prob)
            })
    
    unique_candidates.sort(key=lambda x: x['log_probability'], reverse=True)
    
    return unique_candidates[:top_k]


def predict_and_fill_sentence(model, tokenizer, sentence_plain, device, max_syllables=10, word_to_mask=None):
    """
    Choose a word to mask, then predict the masked word and return the complete sentence.
    
    Args:
        model: The MiniBERT model
        tokenizer: The syllable tokenizer
        sentence_plain: Plain sentence without [MASK]
        device: torch device
        max_syllables: Maximum syllables to predict
        word_to_mask: Optional word to mask. If None, choose randomly
    
    Returns:
        Dictionary with predicted word and complete sentence
    """
    
    words = sentence_plain.split()
    
    if not words:
        return {"error": "Empty sentence"}
    
    # Choose word to mask
    if word_to_mask is None:
        # Choose a random word (avoid very short words)
        valid_words = [(i, w) for i, w in enumerate(words) if len(w) > 2]
        if not valid_words:
            return {"error": "No suitable word to mask"}
        mask_idx, target_word = random.choice(valid_words)
    else:
        # Find the specified word
        try:
            mask_idx = words.index(word_to_mask)
            target_word = word_to_mask
        except ValueError:
            return {"error": f"Word '{word_to_mask}' not found in sentence"}
    
    # Create masked sentence
    context_left = " ".join(words[:mask_idx])
    context_right = " ".join(words[mask_idx + 1:])
    
    # Show what word is being masked and its syllabification
    target_syllables = syllabify_word(target_word)
    print(f"\n   🎯 Masking word: '{target_word}'")
    print(f"   🔤 Syllables: {' + '.join(target_syllables)}")
    
    # Predict the word
    candidates = predict_word_iteratively(
        model, tokenizer,
        context_left=context_left,
        context_right=context_right,
        device=device,
        max_syllables=max_syllables,
        top_k=3
    )
    
    if not candidates:
        return {"error": "No predictions generated"}
    
    # Get best prediction
    best_prediction = candidates[0]
    predicted_word = best_prediction['word']
    
    # Create complete sentence
    if context_left and context_right:
        filled_sentence = f"{context_left} {predicted_word} {context_right}"
    elif context_left:
        filled_sentence = f"{context_left} {predicted_word}"
    else:
        filled_sentence = f"{predicted_word} {context_right}"
    
    return {
        'original_sentence': sentence_plain,
        'target_word': target_word,
        'predicted_word': predicted_word,
        'syllables': best_prediction['syllables'],
        'probability': best_prediction['probability'],
        'filled_sentence': filled_sentence,
        'all_candidates': candidates
    }


def load_quran_examples(filepath="quran-simple-processed.txt", num_examples=100):
    """Load and generate examples from Quranic text file."""
    
    # Read all lines from the file
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    examples = []
    
    # Generate examples from random sentences
    sampled_lines = random.sample(lines, min(num_examples, len(lines)))
    
    for i, sentence in enumerate(sampled_lines):
        # Split into words and filter out very short ones for masking
        words = sentence.split()
        valid_words = [(idx, w) for idx, w in enumerate(words) if len(w) > 2]
        
        if valid_words:
            # Randomly choose a word to mask
            mask_idx, mask_word = random.choice(valid_words)
            examples.append({
                "sentence": sentence,
                "mask_word": mask_word,
                "title": f"Example {i+1}"
            })
    
    return examples


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 80)
    print("🔤 ARABIC MLM INFERENCE - 100 Examples from Quran")
    print("=" * 80)
    print(f"Device: {device}\n")
    
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        model_path="model-bpe/mini_bert_mlm.pt",
        tokenizer_path="syllable_tokenizer.json",
        device=device
    )
    print()
    
    print("=" * 80)
    print("📚 LOADING 100 EXAMPLES FROM QURAN TEXT")
    print("=" * 80)
    
    examples = load_quran_examples("quran-simple-processed.txt", num_examples=100)
    print(f"✓ Loaded {len(examples)} examples\n")
    
    for i, example in enumerate(examples, 1):
        print(f"\n[EXAMPLE {i}] {example['title']}")
        print("-" * 80)
        print(f"📝 INPUT:  {example['sentence']}")
        
        result = predict_and_fill_sentence(
            model, tokenizer, 
            example['sentence'], 
            device, 
            max_syllables=8,
            word_to_mask=example['mask_word']
        )
        
        if 'error' not in result:
            print(f"✅ OUTPUT: {result['filled_sentence']}")
        else:
            print(f"❌ Error: {result['error']}")
    
    print("\n" + "=" * 80)
    print("💬 INTERACTIVE MODE")
    print("=" * 80)
    print("Enter a sentence (it will randomly mask one word)")
    print("Or use format: word=TARGET | sentence (to mask specific word)")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("📝 INPUT:  ").strip()
        
        if user_input.lower() == 'quit':
            print("\n👋 Goodbye!")
            break
        
        if not user_input:
            continue
        
        # Parse optional word specification
        target_word = None
        if '|' in user_input:
            parts = user_input.split('|', 1)
            word_spec = parts[0].strip()
            sentence_text = parts[1].strip()
            if word_spec.startswith('word='):
                target_word = word_spec[5:].strip()
        else:
            sentence_text = user_input
        
        if not sentence_text:
            print("⚠ Please provide a sentence\n")
            continue
        
        result = predict_and_fill_sentence(
            model, tokenizer, 
            sentence_text, 
            device, 
            max_syllables=8,
            word_to_mask=target_word
        )
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}\n")
        else:
            print(f"✅ OUTPUT: {result['filled_sentence']}\n")


if __name__ == "__main__":
    main()

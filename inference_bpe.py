
import torch
import torch.nn as nn
from tokenizers import Tokenizer
import numpy as np
import re
from typing import List, Tuple

from model import MiniBERT
from config import CONFIG

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
        if w in ['[MASK]', '[CLS]', '[SEP]', '[PAD]', '[UNK]']:
            out.append(w)
        else:
            s = syllabify_word(w)
            if s:
                s_no_sukun = [syl.replace(SUKUN, '') for syl in s if syl.replace(SUKUN, '')]
                out.extend(s_no_sukun)
    return " ".join(out)


def load_model_and_tokenizer(model_path="model-bpe/mini_bert_mlm.pt", 
                            tokenizer_path="tokenizer.json", 
                            device="cpu"):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', CONFIG)
    state_dict = checkpoint['model_state_dict']
    vocab_size = state_dict['embedding.token_embedding.weight'].shape[0]
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
    return model, tokenizer, config

def predict_masked_word_syllables(model, tokenizer, target_word, context_left, context_right, 
                                  device, config=None, top_k=5):
    model.eval()
    target_syllables = syllabify_word(target_word)
    if len(target_syllables) == 0:
        return []
    target_syllables_no_sukun = [syl.replace(SUKUN, '') for syl in target_syllables if syl.replace(SUKUN, '')]
    if len(target_syllables_no_sukun) == 0:
        return []
    context_left_syllabified = syllabify_text(context_left) if context_left else ""
    context_right_syllabified = syllabify_text(context_right) if context_right else ""
    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")
    mask_id = tokenizer.token_to_id("[MASK]")
    left_tokens = tokenizer.encode(context_left_syllabified).ids if context_left_syllabified else []
    right_tokens = tokenizer.encode(context_right_syllabified).ids if context_right_syllabified else []
    num_syllables = len(target_syllables_no_sukun)
    tokens = [cls_id] + left_tokens + [mask_id] * num_syllables + right_tokens + [sep_id]
    max_len = config.get('max_len', 256) if config else 256
    if len(tokens) > max_len:
        available_space = max_len - num_syllables - 2
        left_budget = available_space // 2
        right_budget = available_space - left_budget
        left_tokens = left_tokens[-left_budget:] if left_budget > 0 else []
        right_tokens = right_tokens[:right_budget] if right_budget > 0 else []
        tokens = [cls_id] + left_tokens + [mask_id] * num_syllables + right_tokens + [sep_id]
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(input_ids)
        probabilities = torch.softmax(logits, dim=-1)
    mask_positions = [i for i, t in enumerate(tokens) if t == mask_id]
    predictions = []
    for pos in mask_positions:
        pos_probs = probabilities[0, pos]
        candidates = []
        for token_id in torch.argsort(pos_probs, descending=True):
            token_id = token_id.item()
            token_str = tokenizer.id_to_token(token_id)
            prob = pos_probs[token_id].item()
            if token_str in ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]']:
                continue
            candidates.append((token_id, token_str, prob))
            if len(candidates) >= top_k:
                break
        predictions.append(candidates)
    beam_size = min(top_k, 10)
    beam = [([], 0.0)]
    for pos_idx, pos_candidates in enumerate(predictions):
        new_beam = []
        for partial_syls, partial_log_prob in beam:
            for token_id, token_str, prob in pos_candidates:
                new_syls = partial_syls + [token_str]
                new_log_prob = partial_log_prob + np.log(prob)
                new_beam.append((new_syls, new_log_prob))
        new_beam.sort(key=lambda x: x[1], reverse=True)
        beam = new_beam[:beam_size]
    results = []
    seen_words = set()
    for syllables, log_prob in beam:
        recomposed_word = ''.join(syllables)
        if recomposed_word in seen_words:
            continue
        seen_words.add(recomposed_word)
        avg_score = np.exp(log_prob / len(syllables)) if syllables else 0.0
        results.append({
            'word': recomposed_word,
            'syllables': syllables,
            'avg_score': avg_score,
            'log_prob': log_prob
        })
    if not results:
        results.sort(key=lambda x: x['log_prob'], reverse=True)
    return results[:top_k]

def predict_and_fill_sentence(model, tokenizer, sentence_plain, device, config=None, max_syllables=10, word_to_mask=None):
    import random
    words = sentence_plain.split()
    if not words:
        return {"error": "Empty sentence"}
    if word_to_mask is None:
        valid_words = [(i, w) for i, w in enumerate(words) if len(w) > 2]
        if not valid_words:
            return {"error": "No suitable word to mask"}
        mask_idx, target_word = random.choice(valid_words)
    else:
        try:
            mask_idx = words.index(word_to_mask)
            target_word = word_to_mask
        except ValueError:
            return {"error": f"Word '{word_to_mask}' not found in sentence"}
    context_left = " ".join(words[:mask_idx])
    context_right = " ".join(words[mask_idx + 1:])
    target_syllables = syllabify_word(target_word)
    target_syllables_no_sukun = [syl.replace(SUKUN, '') for syl in target_syllables if syl.replace(SUKUN, '')]
    if len(target_syllables_no_sukun) > max_syllables or len(target_syllables_no_sukun) == 0:
        return {"error": f"Word '{target_word}' cannot be tokenized properly"}
    candidates = predict_masked_word_syllables(
        model, tokenizer,
        target_word=target_word,
        context_left=context_left,
        context_right=context_right,
        device=device,
        config=config,
        top_k=5
    )
    if not candidates:
        return {"error": "No predictions generated"}
    best_prediction = candidates[0]
    predicted_word = best_prediction['word']
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
        'avg_score': best_prediction['avg_score'],
        'filled_sentence': filled_sentence,
        'all_candidates': candidates
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, config = load_model_and_tokenizer(
        model_path="model-bpe/mini_bert_mlm.pt",
        tokenizer_path="tokenizer.json",
        device=device
    )
    quran_file = "quran-simple-processed.txt"
    try:
        with open(quran_file, 'r', encoding='utf-8') as f:
            all_lines = [line.strip() for line in f if line.strip()]
        import random
        num_examples = min(100, len(all_lines))
        random_sentences = random.sample(all_lines, num_examples)
    except FileNotFoundError:
        random_sentences = [
            "الحَمدُ لِلّاَهِ رَبِّ العَالَمِينَ الرَّحمَانِ الرَّحِيمِ",
            "إِيَّاكَ نَعبُدُ وَإِيَّاكَ نَستَعِينُ اهدِنَا الصِّرَاطَ المُستَقِيمَ"
        ]
        num_examples = len(random_sentences)
    correct = 0
    total = 0
    for i, sentence in enumerate(random_sentences, 1):
        
        words = sentence.split()
        valid_words = [(idx, w) for idx, w in enumerate(words) if len(w) > 2]
        if not valid_words:
            continue
        mask_idx, target_word = random.choice(valid_words)
        
        result = predict_and_fill_sentence(
            model, tokenizer, 
            sentence, 
            device,
            config=config,
            max_syllables=20,
            word_to_mask=target_word
        )
        if 'error' not in result:
            match = result['predicted_word'] == result['target_word']
            if match:
                correct += 1
            total += 1
            print(f"[{i}/{num_examples}] Target: {result['target_word']} | Predicted: {result['predicted_word']} | Match: {'✓' if match else '✗'}")
        else:
            print(f"[{i}/{num_examples}] Error: {result['error']}")
    if total > 0:
        accuracy = (correct / total) * 100
        print(f"\nAccuracy: {correct}/{total} = {accuracy:.2f}%")


if __name__ == "__main__":
    main()

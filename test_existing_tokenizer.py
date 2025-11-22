import torch
from tokenizers import Tokenizer
from utils import preprocess_arabic_text

def test_tokenizer():
    print("=" * 70)
    print("TESTING EXISTING TOKENIZER FROM CORPUS")
    print("=" * 70)
    
    tokenizer_path = "/home/ubuntu/PycharmProjects/Qwen_aric/bert-cv/raw_data/arabic_syllable_tokenizer_clean/tokenizer.json"
    tokenizer_wordlevel_path = "/home/ubuntu/PycharmProjects/Qwen_aric/bert-cv/raw_data/arabic_syllable_tokenizer_clean/tokenizer.wordlevel.json"
    corpus_path = "/home/ubuntu/PycharmProjects/Qwen_aric/bert-cv/raw_data/arabic_super_cleaned.txt"
    
    print(f"\nTrying word-level tokenizer: {tokenizer_wordlevel_path}")
    try:
        tokenizer = Tokenizer.from_file(tokenizer_wordlevel_path)
        print("✓ Word-level Tokenizer loaded successfully")
        tokenizer_type = "WORDLEVEL"
    except Exception as e:
        print(f"✗ Word-level failed: {e}")
        print(f"\nTrying BPE tokenizer: {tokenizer_path}")
        try:
            tokenizer = Tokenizer.from_file(tokenizer_path)
            print("✓ BPE Tokenizer loaded successfully")
            tokenizer_type = "BPE"
        except Exception as e2:
            print(f"✗ BPE also failed: {e2}\n")
            return
    print()
    
    vocab_size = tokenizer.get_vocab_size()
    print(f"Tokenizer type: {tokenizer_type}")
    print(f"Tokenizer vocab size: {vocab_size}")
    print(f"Special tokens: [CLS], [SEP], [MASK], [UNK], [PAD]\n")
    
    print("=" * 70)
    print("TESTING WITH CORPUS SAMPLES")
    print("=" * 70)
    
    texts = []
    print(f"\nLoading samples from corpus: {corpus_path}")
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 1000:  # Test with 1000 samples
                break
            text = line.strip()
            if len(text) > 0:
                text = preprocess_arabic_text(text)
                if len(text) > 50:
                    texts.append(text)
    
    print(f"✓ Loaded {len(texts)} sample texts\n")
    
    print("=" * 70)
    print("TOKENIZATION TEST")
    print("=" * 70)
    
    total_tokens = 0
    total_unk = 0
    failed_texts = 0
    
    for i, text in enumerate(texts[:50], 1):  # Test first 50 texts
        try:
            encoding = tokenizer.encode(text)
            tokens = encoding.ids
            token_strs = encoding.tokens
            
            if len(tokens) == 0:
                print(f"[{i}] WARNING: Empty tokens for text: {text[:60]}...")
                continue
            
            unk_count = token_strs.count('[UNK]') if token_strs else 0
            total_tokens += len(tokens)
            total_unk += unk_count
            
            if unk_count > 0:
                unk_percentage = (unk_count / len(tokens)) * 100
                print(f"[{i}] {text[:60]}...")
                print(f"     Tokens: {len(tokens)}, [UNK]: {unk_count} ({unk_percentage:.1f}%)")
        except Exception as e:
            failed_texts += 1
            print(f"[{i}] ERROR: {e}")
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    unk_percentage = (total_unk / total_tokens) * 100 if total_tokens > 0 else 0
    
    print(f"Total texts tested: {len(texts[:50])}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Total [UNK] tokens: {total_unk:,}")
    print(f"[UNK] percentage: {unk_percentage:.2f}%")
    print(f"Failed texts: {failed_texts}")
    
    print()
    print("=" * 70)
    print("QUALITY ASSESSMENT")
    print("=" * 70)
    
    if unk_percentage < 5:
        print("✓ EXCELLENT: Very low [UNK] rate, tokenizer works well!")
    elif unk_percentage < 10:
        print("✓ GOOD: Low [UNK] rate, tokenizer works well")
    elif unk_percentage < 15:
        print("⚠ ACCEPTABLE: Moderate [UNK] rate, but usable")
    else:
        print("✗ POOR: High [UNK] rate, may need retraining")
    
    print("\n" + "=" * 70)
    print("DETAILED SAMPLE ANALYSIS")
    print("=" * 70)
    
    # Show first 5 texts with full tokenization
    for i, text in enumerate(texts[:5], 1):
        encoding = tokenizer.encode(text)
        if len(encoding.ids) > 0:
            print(f"\n[Sample {i}]")
            print(f"Text: {text}")
            print(f"Tokens ({len(encoding.tokens)}): {encoding.tokens[:20]}{'...' if len(encoding.tokens) > 20 else ''}")
            print(f"IDs: {encoding.ids[:20]}{'...' if len(encoding.ids) > 20 else ''}")

if __name__ == "__main__":
    test_tokenizer()

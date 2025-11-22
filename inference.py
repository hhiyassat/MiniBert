import torch
import torch.nn as nn
from tokenizers import Tokenizer
import numpy as np

from model import MiniBERT
from utils import preprocess_arabic_text


def merge_subword_tokens(text):
    """
    Merges subword tokens (starting with ##) with the previous word.
    Example: "un ##happy" -> "unhappy"
    """
    if not text:
        return text
    
    tokens = text.split()
    if not tokens:
        return text
    
    merged = [tokens[0]]
    
    for token in tokens[1:]:
        if token.startswith('##'):
            # Remove ## and merge with previous word
            merged[-1] = merged[-1] + token[2:]
        else:
            # Regular token, add as new word
            merged.append(token)
    
    return ' '.join(merged)


def load_model_and_tokenizer(model_path="model/mini_bert_mlm.pt", tokenizer_path="model/tokenizer.json", device="cpu"):
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    print(f"✓ Tokenizer loaded from {tokenizer_path}")
    
    print("Loading model...")
    checkpoint = torch.load(model_path, map_location=device)
    
    config = checkpoint.get('config', {})
    
    # Get vocab_size from actual model weights (not from config which might be wrong)
    state_dict = checkpoint['model_state_dict']
    vocab_size = state_dict['embedding.token_embedding.weight'].shape[0]
    
    model = MiniBERT(
        vocab_size=vocab_size,
        hidden_dim=config.get('hidden_dim', 256),
        num_layers=config.get('num_layers', 4),
        num_heads=config.get('num_heads', 4),
        intermediate_dim=config.get('intermediate_dim', 1024),
        max_len=config.get('max_len', 128),
        dropout=config.get('dropout', 0.1)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded from {model_path}")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Hidden dimension: {config.get('hidden_dim', 256)}")
    print(f"  Number of layers: {config.get('num_layers', 4)}")
    if 'epoch' in checkpoint:
        print(f"  Training epoch: {checkpoint['epoch'] + 1}")
    if 'loss' in checkpoint:
        print(f"  Training loss: {checkpoint['loss']:.4f}")
    print(f"  Device: {device}\n")
    
    return model, tokenizer


def predict_masked_sentence(model, tokenizer, text, device, top_k=5):
    model.eval()
    
    text = preprocess_arabic_text(text)
    
    encoding = tokenizer.encode(text)
    tokens = encoding.ids
    token_strs = encoding.tokens
    
    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")
    mask_id = tokenizer.token_to_id("[MASK]")
    
    tokens = [cls_id] + tokens + [sep_id]
    token_strs = ["[CLS]"] + token_strs + ["[SEP]"]
    
    mask_positions = [i for i, token in enumerate(tokens) if token == mask_id]
    
    if not mask_positions:
        return {"error": "No [MASK] tokens found in the input text"}
    
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
    
    with torch.no_grad():
        logits = model(input_ids)
        probabilities = torch.softmax(logits, dim=-1)
    
    predictions = {}
    for idx, mask_pos in enumerate(mask_positions):
        top_k_probs, top_k_indices = torch.topk(probabilities[0, mask_pos], top_k)
        
        top_k_tokens = []
        for prob, token_id in zip(top_k_probs, top_k_indices):
            token = tokenizer.id_to_token(token_id.item())
            top_k_tokens.append({
                'token': token,
                'probability': prob.item()
            })
        
        predictions[f"mask_{idx+1}"] = {
            'position': mask_pos,
            'top_predictions': top_k_tokens
        }
    
    return {
        'input_text': text,
        'tokenized': token_strs,
        'predictions': predictions
    }


def fill_mask(model, tokenizer, text, device):
    model.eval()
    
    text = preprocess_arabic_text(text)
    
    encoding = tokenizer.encode(text)
    tokens = encoding.ids
    
    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")
    mask_id = tokenizer.token_to_id("[MASK]")
    
    tokens = [cls_id] + tokens + [sep_id]
    
    mask_positions = [i for i, token in enumerate(tokens) if token == mask_id]
    
    if not mask_positions:
        return text
    
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
    
    with torch.no_grad():
        logits = model(input_ids)
        predictions = torch.argmax(logits, dim=-1)[0]
    
    filled_tokens = tokens.copy()
    for mask_pos in mask_positions:
        filled_tokens[mask_pos] = predictions[mask_pos].item()
    
    filled_text = tokenizer.decode(filled_tokens[1:-1])
    
    # Merge subword tokens (## prefix) with previous words
    filled_text = merge_subword_tokens(filled_text)
    
    return filled_text


def batch_predict(model, tokenizer, texts, device, top_k=3):
    results = []
    
    print(f"Processing {len(texts)} texts...\n")
    for i, text in enumerate(texts, 1):
        print(f"[{i}/{len(texts)}] {text}")
        result = predict_masked_sentence(model, tokenizer, text, device, top_k)
        results.append(result)
        
        if 'predictions' in result:
            for mask_key, pred_info in result['predictions'].items():
                top_pred = pred_info['top_predictions'][0]
                print(f"  → {mask_key}: {top_pred['token']} ({top_pred['probability']:.4f})")
        print()
    
    return results


def main():
    device = torch.device("cpu")
    print(f"Using device: {device}\n")
    
    print("=" * 70)
    print("LOADING MODEL AND TOKENIZER")
    print("=" * 70)
    model, tokenizer = load_model_and_tokenizer(
 
        device=device
    )
    
    print("=" * 70)
    print("EXAMPLE 1: رجل حلف علي سلعه... (2 masks)")
    print("=" * 70)
    text1 = "رجل حلف علي سلعه [MASK] العصر أخذها بكذا وكذا فصدقه الذي باعه فأخذها [MASK] كاذب"
    expected1_mask1 = "بعد"
    expected1_mask2 = "وهو"
    reference1 = "رجل حلف علي سلعه بعد العصر أخذها بكذا وكذا فصدقه الذي باعه فأخذها وهو كاذب"
    result1 = predict_masked_sentence(model, tokenizer, text1, device, top_k=5)
    print(f"Reference: {reference1}")
    print(f"Expected words: **{expected1_mask1}** (mask_1), **{expected1_mask2}** (mask_2)\n")
    print(f"Predictions:\n")
    
    mask_counter = 0
    expected_words = [expected1_mask1, expected1_mask2]
    temp_text = text1
    
    for mask_key, pred_info in result1['predictions'].items():
        expected_word = expected_words[mask_counter]
        print(f"{mask_key} - Expected: **{expected_word}**")
        for i, pred in enumerate(pred_info['top_predictions'], 1):
            # Get the token, removing ## if it exists for display
            display_token = pred['token']
            if display_token.startswith('##'):
                display_token = display_token[2:]
            
            filled_sentence = temp_text.replace('[MASK]', pred['token'], 1)
            # Merge subword tokens in the filled sentence
            filled_sentence = merge_subword_tokens(filled_sentence)
            
            match = "✓" if pred['token'] == expected_word or display_token == expected_word else " "
            print(f"  {match} {i}. {display_token:15s} ({pred['probability']:.4f})")
            print(f"     → {filled_sentence}")
        
        # Use the token (with ## if needed) for next replacement
        best_token = pred_info['top_predictions'][0]['token']
        temp_text = temp_text.replace('[MASK]', best_token, 1)
        temp_text = merge_subword_tokens(temp_text)
        mask_counter += 1
        print()
    
    print("=" * 70)
    print("EXAMPLE 2: ملك كذاب...")
    print("=" * 70)
    text2 = "ملك كذاب [MASK] مستكبر"
    expected2 = "وعايل"
    reference2 = text2.replace('[MASK]', expected2)
    result2 = predict_masked_sentence(model, tokenizer, text2, device, top_k=5)
    print(f"Reference: {reference2}")
    print(f"Expected word: **{expected2}**\n")
    print(f"Predictions:\n")
    for mask_key, pred_info in result2['predictions'].items():
        for i, pred in enumerate(pred_info['top_predictions'], 1):
            # Get the token, removing ## if it exists for display
            display_token = pred['token']
            if display_token.startswith('##'):
                display_token = display_token[2:]
            
            filled_sentence = text2.replace('[MASK]', pred['token'], 1)
            # Merge subword tokens in the filled sentence
            filled_sentence = merge_subword_tokens(filled_sentence)
            
            match = "✓" if pred['token'] == expected2 or display_token == expected2 else " "
            print(f"  {match} {i}. {display_token:15s} ({pred['probability']:.4f})")
            print(f"     → {filled_sentence}")
    print()
    
    print("=" * 70)
    print("EXAMPLE 3: فهذه الاثار كلها...")
    print("=" * 70)
    text3 = "فهذه الاثار كلها [MASK] المعاني"
    expected3 = "مكشوفه"
    reference3 = text3.replace('[MASK]', expected3)
    result3 = predict_masked_sentence(model, tokenizer, text3, device, top_k=5)
    print(f"Reference: {reference3}")
    print(f"Expected word: **{expected3}**\n")
    print(f"Predictions:\n")
    for mask_key, pred_info in result3['predictions'].items():
        for i, pred in enumerate(pred_info['top_predictions'], 1):
            # Get the token, removing ## if it exists for display
            display_token = pred['token']
            if display_token.startswith('##'):
                display_token = display_token[2:]
            
            filled_sentence = text3.replace('[MASK]', pred['token'], 1)
            # Merge subword tokens in the filled sentence
            filled_sentence = merge_subword_tokens(filled_sentence)
            
            match = "✓" if pred['token'] == expected3 or display_token == expected3 else " "
            print(f"  {match} {i}. {display_token:15s} ({pred['probability']:.4f})")
            print(f"     → {filled_sentence}")
    print()
    
    print("=" * 70)
    print("EXAMPLE 4: تصحيح هذا الحديث...")
    print("=" * 70)
    text4 = "تصحيح هذا الحديث وما ذكرنا [MASK] من الاحاديث في هذا الباب"
    expected4 = "قبله"
    reference4 = text4.replace('[MASK]', expected4)
    result4 = predict_masked_sentence(model, tokenizer, text4, device, top_k=5)
    print(f"Reference: {reference4}")
    print(f"Expected word: **{expected4}**\n")
    print(f"Predictions:\n")
    for mask_key, pred_info in result4['predictions'].items():
        for i, pred in enumerate(pred_info['top_predictions'], 1):
            # Get the token, removing ## if it exists for display
            display_token = pred['token']
            if display_token.startswith('##'):
                display_token = display_token[2:]
            
            filled_sentence = text4.replace('[MASK]', pred['token'], 1)
            # Merge subword tokens in the filled sentence
            filled_sentence = merge_subword_tokens(filled_sentence)
            
            match = "✓" if pred['token'] == expected4 or display_token == expected4 else " "
            print(f"  {match} {i}. {display_token:15s} ({pred['probability']:.4f})")
            print(f"     → {filled_sentence}")
    print()
    
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Interactive Mode")
    print("=" * 70)
    print("Enter your own sentences with [MASK] tokens!")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("Enter text (with [MASK]): ").strip()
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        if "[MASK]" not in user_input:
            print("⚠ No [MASK] token found. Please include at least one [MASK]\n")
            continue
        
        result = predict_masked_sentence(model, tokenizer, user_input, device, top_k=5)
        
        if 'error' in result:
            print(f"⚠ {result['error']}\n")
            continue
        
        print("\nTop predictions:")
        for mask_key, pred_info in result['predictions'].items():
            print(f"\n{mask_key}:")
            temp_input = user_input
            for i, pred in enumerate(pred_info['top_predictions'], 1):
                # Get the token, removing ## if it exists for display
                display_token = pred['token']
                if display_token.startswith('##'):
                    display_token = display_token[2:]
                
                filled_sentence = temp_input.replace('[MASK]', pred['token'], 1)
                # Merge subword tokens in the filled sentence
                filled_sentence = merge_subword_tokens(filled_sentence)
                
                print(f"  {i}. {display_token:15s} ({pred['probability']:.4f})")
                print(f"     → {filled_sentence}")
        
        filled = fill_mask(model, tokenizer, user_input, device)
        print(f"\nBest filled sentence: {filled}\n")
        print("-" * 70)


if __name__ == "__main__":
    main()

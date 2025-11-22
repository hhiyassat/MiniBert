from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace


def train_tokenizer(texts, vocab_size=8000):
    print("Training WordPiece tokenizer...")
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    
    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        min_frequency=3,
        show_progress=True
    )
    
    tokenizer.train_from_iterator(texts, trainer)
    print(f"✓ Tokenizer trained with vocabulary size: {tokenizer.get_vocab_size()}")
    return tokenizer

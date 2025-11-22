# Process All Files

This folder processes all 341 Arabic text files from the Tashkeela corpus.

## Files:
- `arabic_utils.py` - Utility functions for text processing
- `process_arabic_texts.py` - Main script to process all files

## Usage:

```bash
python process_arabic_texts.py
```

The script will:
1. Ask if you want to remove all diacritics or keep them
2. Process all 341 text files recursively
3. Combine them into one output file
4. Remove sukun from lam (ل) in all cases

## Output:
- `combined_arabic_text.txt` - Combined processed text (~600 MB)

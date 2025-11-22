# Test 1 File

This folder tests the Arabic text processing on a single file (aljazeera.txt).

## Files:
- `arabic_utils.py` - Utility functions for text processing
- `1_test_file.py` - Test script for single file processing

## Usage:

```bash
python 1_test_file.py
```

The script will:
1. Ask if you want to remove all diacritics or keep them
2. Process only the aljazeera.txt file
3. Show before/after comparison of text processing
4. Save output to `1_test_output.txt`
5. Remove sukun from lam (ل) in all cases

## Output:
- `name_of_file.txt` - Processed aljazeera.txt file (~400 KB)

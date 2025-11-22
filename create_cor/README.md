# Arabic Text Processing - Create Corpus

This project processes Arabic text files from the Tashkeela corpus, with options to remove diacritics or keep them while removing sukun from lam.

## Project Structure:

```
create_cor/
├── process_all_files/          # Process all 341 files
│   ├── arabic_utils.py
│   ├── process_arabic_texts.py
│   └── README.md
│
├── test_1_file/                # Test on single file
│   ├── arabic_utils.py
│   ├── 1_test_file.py
│   └── README.md
│
└── README.md (this file)
└── config.py
```

## Tasks:

### 1. Process All Files
Navigate to `process_all_files/` and run:
```bash
python process_arabic_texts.py
```

This processes all 341 text files from the Tashkeela corpus and combines them into one file.

### 2. Test Single File
Navigate to `test_1_file/` and run:
```bash
python 1_test_file.py
```

This tests the processing on a single file (aljazeera.txt) with before/after comparison.

## Features:

✅ Remove all Arabic diacritics (optional)
✅ Keep diacritics but remove sukun from lam (ل)
✅ Recursive file scanning
✅ Error handling
✅ Progress tracking
✅ File size reporting

## Configuration:

both folders has use `config.py` with:
- Source directories
- Output directories
- File names and paths

Edit `config.py` in each folder to change paths or settings.

## Utility Functions:

All core functions are in `arabic_utils.py`:
- `remove_all_diacritics()` - Remove all Arabic diacritics
- `remove_lam_sukun()` - Remove sukun from lam only
- `process_text()` - Main processing function
- `read_file()` - File reading with error handling
- `write_file()` - File writing with directory creation
- `find_all_txt_files()` - Recursive file finding
- `find_specific_file()` - Find a specific file
- `get_file_size_kb()` / `get_file_size_mb()` - File size utilities

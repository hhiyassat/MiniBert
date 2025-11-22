import re
from pathlib import Path


ARABIC_DIACRITICS = [
    '\u064B', '\u064C', '\u064D', '\u064E', '\u064F',
    '\u0650', '\u0651', '\u0652', '\u0653', '\u0654',
    '\u0655', '\u0656', '\u0657', '\u0658', '\u0670',
]

SUKUN = '\u0652'
LAM = 'ل'

DIACRITICS_PATTERN = re.compile('|'.join(map(re.escape, ARABIC_DIACRITICS)))
LAM_SUKUN_PATTERN = re.compile(f'{LAM}{SUKUN}')

# Arabic Unicode ranges
ARABIC_LETTERS = re.compile(r'[\u0600-\u06FF]')  # Arabic block
# Pattern to keep: Arabic letters, diacritics, and allowed punctuation
ALLOWED_CHARS = re.compile(r'[\u0600-\u06FF\s?!.,،؛]')  # Arabic, space, and punctuation


def remove_all_diacritics(text):
    return DIACRITICS_PATTERN.sub('', text)


def remove_lam_sukun(text):
    return LAM_SUKUN_PATTERN.sub(LAM, text)


def remove_non_arabic(text, keep_diacritics=False):
    """Remove all non-Arabic characters except space and allowed punctuation"""
    if keep_diacritics:
        # Keep Arabic letters, diacritics, numbers, spaces, and allowed punctuation
        result = ''.join(c for c in text if re.match(r'[\u0600-\u06FF\s?!.,،؛\u064B-\u0658\u0670]', c))
    else:
        # Keep only Arabic letters, spaces, and allowed punctuation
        result = ''.join(c for c in text if re.match(r'[\u0600-\u06FF\s?!.,،؛]', c))
    return result


def process_text(text, remove_diacritics=False):
    text = remove_lam_sukun(text)
    # Remove non-Arabic characters (keep diacritics only if not removing them)
    text = remove_non_arabic(text, keep_diacritics=not remove_diacritics)
    if remove_diacritics:
        text = remove_all_diacritics(text)
    return text


def remove_empty_lines(text):
    lines = text.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    return '\n'.join(non_empty_lines)


def read_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""


def write_file(file_path, content):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)


def find_all_txt_files(directory):
    return sorted(Path(directory).rglob('*.txt'))


def find_specific_file(directory, filename):
    file_path = Path(directory) / filename
    return [file_path] if file_path.exists() else []


def get_file_size_kb(file_path):
    return Path(file_path).stat().st_size / 1024


def get_file_size_mb(file_path):
    return Path(file_path).stat().st_size / (1024 * 1024)

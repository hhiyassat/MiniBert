import re


def clean_arabic_text(text):
    text = re.sub(r'###Human:\s*', '', text)
    text = re.sub(r'###Assistant:\s*', '', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = text.strip()
    return text


def normalize_arabic(text):
    text = text.replace('أ', 'ا')
    text = text.replace('إ', 'ا')
    text = text.replace('آ', 'ا')
    text = text.replace('ٱ', 'ا')
    arabic_diacritics = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = arabic_diacritics.sub('', text)
    return text


def preprocess_arabic_text(text):
    text = clean_arabic_text(text)
    text = normalize_arabic(text)
    return text

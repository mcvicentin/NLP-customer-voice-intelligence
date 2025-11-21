# src/data_processing/text_cleaning.py

import re
import contractions
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

stopwords_en = set(ENGLISH_STOP_WORDS)


def clean_text(text: str) -> str:
    """Basic text normalization."""
    text = contractions.fix(text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text: str) -> str:
    """Remove stopwords + tokens of size 1."""
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords_en and len(t) > 1]
    return " ".join(tokens)


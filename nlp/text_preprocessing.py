"""Phase 3: Text Preprocessing & Normalization (100% Pure Python + Scikit-learn)."""

import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def clean_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[•●▪▸►◦]", "- ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _light_stem(word: str) -> str:
    """Lightweight suffix normalization without external C-extensions."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > 4 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def preprocess_for_tfidf(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+|\S+@\S+", " ", text)
    text = re.sub(r"[^a-z0-9\+\#\.\-\s]", " ", text)
    tokens = [
        _light_stem(tok)
        for tok in text.split()
        if tok not in ENGLISH_STOP_WORDS and len(tok) > 1
    ]
    return " ".join(tokens)
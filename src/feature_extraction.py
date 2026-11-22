"""Faithful reproduction of the model author's public 0.2.4 extractor."""

from __future__ import annotations

import re
import string
from typing import Any

FEATURES = [
    "char_count",
    "word_count",
    "avg_word_len",
    "punct_count",
    "sentence_count",
    "avg_sentence_len",
    "upper_case_count",
    "title_case_count",
]


def extract_stylometric_features(text: Any) -> dict[str, float | int]:
    """Extract features with the maintainer's whitespace/regex semantics.

    Non-string and whitespace-only values produce zeros, as in the published
    package. The RAID loader excludes those records from the evaluation sample.
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "char_count": 0,
            "word_count": 0,
            "avg_word_len": 0.0,
            "punct_count": 0,
            "sentence_count": 0,
            "avg_sentence_len": 0.0,
            "upper_case_count": 0,
            "title_case_count": 0,
        }

    words = text.split()
    word_count = len(words)
    sentences = re.split(r"[.!?]+", text)
    sentence_count = len([sentence for sentence in sentences if sentence.strip()])
    sentence_count = sentence_count if sentence_count > 0 else 1

    return {
        "char_count": len(text),
        "word_count": word_count,
        "avg_word_len": sum(len(word) for word in words) / word_count if word_count else 0.0,
        "punct_count": sum(1 for char in text if char in string.punctuation),
        "sentence_count": sentence_count,
        "avg_sentence_len": word_count / sentence_count,
        "upper_case_count": sum(1 for word in words if word.isupper() and word.isalpha()),
        "title_case_count": sum(1 for word in words if word.istitle()),
    }

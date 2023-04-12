"""Tokenize data used AutoTokenizer."""
from transformers import AutoTokenizer
from typing import List, Dict


def tokenize_text(
        text: List[str],
        model_name: str,
) -> Dict:
    """Tokenize text used AutoTokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=15  # max_length=10
    )
    return inputs

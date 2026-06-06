"""Load raw datasets and produce unified word-level (tokens, labels) records."""
from __future__ import annotations
import ast
from pathlib import Path

import pandas as pd

from src.data.label_mapping import map_labels


# ── Kaggle PII Data Detection ────────────────────────────────────────────────

def load_kaggle_dataset(filepath: str | Path) -> list[dict]:
    """Load the Kaggle student-essay PII dataset.

    Returns records with word-level tokens, unified BIO labels, and
    trailing_whitespace booleans for faithful text reconstruction.
    """
    df = pd.read_csv(filepath, dtype=str)
    records = []
    skipped = 0

    for _, row in df.iterrows():
        try:
            tokens: list[str] = ast.literal_eval(row["tokens"])
            raw_labels: list[str] = ast.literal_eval(row["labels"])
            trailing: list[bool] = ast.literal_eval(row["trailing_whitespace"])
        except Exception:
            skipped += 1
            continue

        if len(tokens) != len(raw_labels):
            skipped += 1
            continue

        unified = map_labels(raw_labels, "kaggle")
        records.append({
            "tokens": tokens,
            "labels": unified,
            "trailing_whitespace": trailing,
            "source": "kaggle",
        })

    if skipped:
        print(f"[kaggle] Skipped {skipped} malformed rows.")
    return records


# ── ai4privacy PII Masking ────────────────────────────────────────────────────

def _reconstruct_from_wordpiece(
    wp_tokens: list[str], wp_labels: list[str]
) -> tuple[list[str], list[str]]:
    """Merge WordPiece ## continuations back into whole words.

    The first sub-word of each word keeps its label; ## continuations are
    appended to the previous word and discarded from the label list.
    """
    words: list[str] = []
    word_labels: list[str] = []
    for token, label in zip(wp_tokens, wp_labels):
        if token.startswith("##"):
            if words:
                words[-1] += token[2:]
        else:
            words.append(token)
            word_labels.append(label)
    return words, word_labels


def load_ai4privacy_dataset(filepath: str | Path) -> list[dict]:
    """Load the ai4privacy PII masking dataset (PII43k.csv).

    Column layout:
        0: Template
        1: Filled Template
        2: Tokenised Filled Template  ← WordPiece tokens
        3: Tokens                     ← BIO labels (confusingly named)
    """
    # Some rows have unescaped commas inside fields → use Python engine + skip bad lines
    try:
        df = pd.read_csv(filepath, dtype=str, on_bad_lines="skip", engine="python")
    except TypeError:
        # pandas < 1.3 fallback
        df = pd.read_csv(filepath, dtype=str, error_bad_lines=False, engine="python")

    records = []
    skipped = 0

    for _, row in df.iterrows():
        try:
            wp_tokens: list[str] = ast.literal_eval(row["Tokenised Filled Template"])
            wp_labels: list[str] = ast.literal_eval(row["Tokens"])
        except Exception:
            skipped += 1
            continue

        if len(wp_tokens) != len(wp_labels):
            skipped += 1
            continue

        tokens, raw_labels = _reconstruct_from_wordpiece(wp_tokens, wp_labels)
        unified = map_labels(raw_labels, "ai4privacy")
        records.append({
            "tokens": tokens,
            "labels": unified,
            "source": "ai4privacy",
        })

    if skipped:
        print(f"[ai4privacy] Skipped {skipped} malformed rows.")
    return records

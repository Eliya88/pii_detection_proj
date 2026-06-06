"""File I/O utilities and text-reconstruction helpers."""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ── JSON Lines ───────────────────────────────────────────────────────────────

def save_jsonl(records: list[dict], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path: Path | str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ── JSON ─────────────────────────────────────────────────────────────────────

def save_json(obj: dict, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)


def load_json(path: Path | str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Text reconstruction ───────────────────────────────────────────────────────

def tokens_to_text(tokens: list[str], trailing_whitespace: list[bool] | None = None) -> str:
    """Rebuild plain text from a token list.

    If trailing_whitespace is provided (Kaggle format), it controls spacing.
    Otherwise tokens are joined with single spaces.
    """
    if trailing_whitespace is None:
        return " ".join(tokens)
    text = ""
    for token, ws in zip(tokens, trailing_whitespace):
        text += token
        if ws:
            text += " "
    return text


def tokens_to_text_with_offsets(
    tokens: list[str],
    trailing_whitespace: list[bool] | None = None,
) -> tuple[str, list[int]]:
    """Returns (text, char_starts) where char_starts[i] is the start char of token i."""
    text = ""
    starts: list[int] = []
    for i, token in enumerate(tokens):
        starts.append(len(text))
        text += token
        if trailing_whitespace is not None:
            if trailing_whitespace[i]:
                text += " "
        else:
            if i < len(tokens) - 1:
                text += " "
    return text, starts


# ── Span → BIO conversion ────────────────────────────────────────────────────

def char_spans_to_bio(
    tokens: list[str],
    token_char_starts: list[int],
    spans: list[dict],
) -> list[str]:
    """Convert character-level entity spans to a BIO label list.

    spans: [{"type": "PERSON", "start": 0, "end": 10}, ...]
    Handles partial overlaps: a token is included if it overlaps the span.
    """
    labels = ["O"] * len(tokens)
    token_char_ends = [token_char_starts[i] + len(tokens[i]) for i in range(len(tokens))]

    for span in sorted(spans, key=lambda s: s["start"]):
        entity = span["type"]
        s_start, s_end = span["start"], span["end"]

        span_toks = [
            i for i, (ts, te) in enumerate(zip(token_char_starts, token_char_ends))
            if ts < s_end and te > s_start
        ]

        for j, tok_idx in enumerate(span_toks):
            labels[tok_idx] = f"{'B' if j == 0 else 'I'}-{entity}"

    return labels

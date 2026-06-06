"""Stratified train / val / test split at the document level."""
from __future__ import annotations
import random


def _has_pii(record: dict) -> bool:
    return any(l != "O" for l in record["labels"])


def stratified_split(
    records: list[dict],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split records into train / val / test while preserving PII density.

    Stratification is binary: records with at least one PII token vs. all-O.
    """
    rng = random.Random(seed)

    pii_recs  = [r for r in records if _has_pii(r)]
    none_recs = [r for r in records if not _has_pii(r)]

    rng.shuffle(pii_recs)
    rng.shuffle(none_recs)

    def _cut(lst: list) -> tuple[list, list, list]:
        n = len(lst)
        t = int(n * train_ratio)
        v = int(n * val_ratio)
        return lst[:t], lst[t:t+v], lst[t+v:]

    p_tr, p_va, p_te = _cut(pii_recs)
    n_tr, n_va, n_te = _cut(none_recs)

    train = p_tr + n_tr
    val   = p_va + n_va
    test  = p_te + n_te

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    print(
        f"Split → train: {len(train)}, val: {len(val)}, test: {len(test)}\n"
        f"  (PII records — train: {len(p_tr)}, val: {len(p_va)}, test: {len(p_te)})"
    )
    return train, val, test

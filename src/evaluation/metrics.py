"""seqeval-based entity-level evaluation."""
from __future__ import annotations

from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics_from_sequences(
    true_labels: list[list[str]],
    pred_labels: list[list[str]],
) -> dict:
    """Compute entity-level P / R / F1 (macro) plus per-entity breakdown."""
    assert len(true_labels) == len(pred_labels), "Length mismatch."

    # Trim predictions to true label length (handles truncation edge cases)
    trimmed_preds = [
        p[: len(t)] for t, p in zip(true_labels, pred_labels)
    ]
    # Pad predictions that are shorter than ground truth
    padded_preds = [
        p + ["O"] * (len(t) - len(p)) for t, p in zip(true_labels, trimmed_preds)
    ]

    report = classification_report(
        true_labels, padded_preds, output_dict=True, zero_division=0
    )

    return {
        "precision": round(precision_score(true_labels, padded_preds, zero_division=0), 4),
        "recall":    round(recall_score(true_labels, padded_preds, zero_division=0), 4),
        "f1":        round(f1_score(true_labels, padded_preds, zero_division=0), 4),
        "per_entity": {
            k: {
                "precision": round(v["precision"], 4),
                "recall":    round(v["recall"], 4),
                "f1":        round(v["f1-score"], 4),
                "support":   v["support"],
            }
            for k, v in report.items()
            if isinstance(v, dict) and k not in ("micro avg", "macro avg", "weighted avg")
        },
    }

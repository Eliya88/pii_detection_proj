"""Presidio-based PII detection (Model 1 – REGEX / Rule-based)."""
from __future__ import annotations
import re
import sys
import time
from pathlib import Path

_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}")

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

from src.models.regex_presidio.custom_recognizers import get_custom_recognizers
from src.utils.io import (
    load_jsonl,
    save_json,
    tokens_to_text_with_offsets,
    char_spans_to_bio,
)
from src.evaluation.metrics import compute_metrics_from_sequences

# Presidio entity type → unified label
PRESIDIO_TO_UNIFIED: dict[str, str | None] = {
    "PERSON":           "PERSON",
    "EMAIL_ADDRESS":    "EMAIL",
    "PHONE_NUMBER":     "PHONE",
    "LOCATION":         "ADDRESS",
    "URL":              "URL",
    "US_SSN":           "ID",
    "US_DRIVER_LICENSE":"ID",
    "US_PASSPORT":      "ID",
    "CREDIT_CARD":      "ID",
    "IBAN_CODE":        "ID",
    "US_BANK_NUMBER":   "ID",
    "IP_ADDRESS":       "ID",
    "SG_NRIC_FIN":      "ID",
    "AU_ACN":           "ID",
    "AU_TFN":           "ID",
    "IN_PAN":           "ID",
    "AU_MEDICARE":      "ID",
    "UK_NHS":           "ID",
    "ES_NIF":           "ID",
    "IT_FISCAL_CODE":   "ID",
    "IT_DRIVER_LICENSE":"ID",
    "IT_VAT_CODE":      "ID",
    "IT_PASSPORT":      "ID",
    "IT_IDENTITY_CARD": "ID",
    "PL_PESEL":         "ID",
    "US_ITIN":          "ID",
    "MEDICAL_LICENSE":  "ID",
    "USERNAME":         "USERNAME",
    "DATE_TIME":        None,
    "NRP":              None,
    "CRYPTO":           None,
}


def build_analyzer() -> AnalyzerEngine:
    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
    })
    nlp_engine = provider.create_engine()
    try:
        analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
    except TypeError:
        analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
    # Remove the CryptoRecognizer — it crashes on some text in this Presidio version
    analyzer.registry.recognizers = [
        r for r in analyzer.registry.recognizers
        if "crypto" not in type(r).__name__.lower()
    ]
    for rec in get_custom_recognizers():
        analyzer.registry.add_recognizer(rec)
    return analyzer


def predict_record(
    record: dict,
    analyzer: AnalyzerEngine,
    score_threshold: float = 0.35,
) -> list[str]:
    """Run Presidio on one record and return BIO labels aligned to its tokens."""
    tokens = record["tokens"]
    trailing_ws = record.get("trailing_whitespace")
    text, char_starts = tokens_to_text_with_offsets(tokens, trailing_ws)

    results = analyzer.analyze(text=text, language="en", score_threshold=score_threshold)

    spans = []
    for r in results:
        unified = PRESIDIO_TO_UNIFIED.get(r.entity_type)
        if unified:
            spans.append({"type": unified, "start": r.start, "end": r.end})

    # Presidio's built-in EMAIL_ADDRESS recognizer misses many emails (its URL
    # recognizer catches the domain part instead). Run our own email regex on
    # the full text and add EMAIL spans.
    email_spans = []
    for m in _EMAIL_RE.finditer(text):
        email_spans.append({"type": "EMAIL", "start": m.start(), "end": m.end()})

    # Remove URL/PERSON spans that overlap with detected emails
    if email_spans:
        def _overlaps_email(s):
            return any(s["start"] < e["end"] and s["end"] > e["start"] for e in email_spans)
        spans = [s for s in spans if not _overlaps_email(s)]
        spans.extend(email_spans)

    return char_spans_to_bio(tokens, char_starts, spans)


def run_presidio_pipeline(
    test_path: str | Path,
    output_path: str | Path,
    score_threshold: float = 0.35,
) -> dict:
    records = load_jsonl(test_path)
    print(f"Running Presidio on {len(records)} records …")

    analyzer = build_analyzer()
    preds: list[list[str]] = []
    t0 = time.perf_counter()

    for rec in records:
        preds.append(predict_record(rec, analyzer, score_threshold))

    elapsed = time.perf_counter() - t0
    latency_ms = (elapsed / len(records)) * 1000

    true_labels = [r["labels"] for r in records]
    metrics = compute_metrics_from_sequences(true_labels, preds)
    metrics["latency_ms_per_record"] = round(latency_ms, 3)

    result = {"metrics": metrics, "predictions": preds, "model": "presidio"}
    save_json(result, output_path)
    print(
        f"Presidio → F1={metrics['f1']:.4f}  P={metrics['precision']:.4f}"
        f"  R={metrics['recall']:.4f}  latency={latency_ms:.1f} ms/rec"
    )
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",      default="data/processed/test.jsonl")
    parser.add_argument("--output",    default="outputs/results/regex_results.json")
    parser.add_argument("--threshold", type=float, default=0.35)
    args = parser.parse_args()
    run_presidio_pipeline(args.test, args.output, args.threshold)

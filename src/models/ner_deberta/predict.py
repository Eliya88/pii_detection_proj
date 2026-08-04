"""Batch inference with a fine-tuned DeBERTa token-classifier."""
from __future__ import annotations
import sys
import time
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification

from src.data.label_mapping import LABEL2ID, ID2LABEL, UNIFIED_LABELS
from src.models.ner_deberta.dataset import PIIDataset
from src.utils.io import load_jsonl, save_jsonl, save_json


def predict(
    records: list[dict],
    model_dir: str | Path,
    batch_size: int = 32,
    max_length: int = 512,
    device: str | None = None,
) -> list[list[str]]:
    """Return predicted BIO label sequences (word-level) for each record."""
    model_dir = Path(model_dir)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    dataset = PIIDataset(records, tokenizer, LABEL2ID, max_length)
    collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)

    all_preds: list[list[str]] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch   = batch["labels"]            # still on CPU

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits  = outputs.logits.cpu().numpy()
            pred_ids = np.argmax(logits, axis=2)

            for pred_row, label_row in zip(pred_ids, labels_batch.numpy()):
                seq: list[str] = []
                for pred_id, label_id in zip(pred_row, label_row):
                    if label_id == -100:
                        continue
                    seq.append(ID2LABEL[int(pred_id)])
                all_preds.append(seq)

    return all_preds


def predict_and_save(
    test_path: str | Path,
    model_dir: str | Path,
    output_path: str | Path,
    batch_size: int = 32,
    max_length: int = 512,
) -> dict:
    from src.evaluation.metrics import compute_metrics_from_sequences

    records = load_jsonl(test_path)

    t0 = time.perf_counter()
    preds   = predict(records, model_dir, batch_size=batch_size, max_length=max_length)
    elapsed = time.perf_counter() - t0
    latency_ms = (elapsed / len(records)) * 1000

    true_labels = [r["labels"] for r in records]
    metrics     = compute_metrics_from_sequences(true_labels, preds)
    metrics["latency_ms_per_record"] = round(latency_ms, 3)

    result = {"metrics": metrics, "predictions": preds}
    save_json(result, output_path)
    print(f"Results saved → {output_path}")
    print(f"  F1={metrics['f1']:.4f}  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  latency={latency_ms:.1f} ms/rec")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",   default="data/processed/test.jsonl")
    parser.add_argument("--model",  default="outputs/checkpoints/deberta/best_model")
    parser.add_argument("--output", default="outputs/results/ner_results.json")
    parser.add_argument("--batch",  type=int, default=32)
    args = parser.parse_args()
    predict_and_save(args.test, args.model, args.output, batch_size=args.batch)

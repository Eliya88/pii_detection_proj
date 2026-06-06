"""Fine-tune DeBERTa-v3-base for PII token classification."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)
from seqeval.metrics import f1_score, precision_score, recall_score

from src.data.label_mapping import LABEL2ID, ID2LABEL, UNIFIED_LABELS
from src.models.ner_deberta.dataset import PIIDataset
from src.utils.io import load_jsonl, save_json


def compute_metrics(p):
    logits, labels = p
    predictions = np.argmax(logits, axis=2)

    true_labels, true_preds = [], []
    for pred_row, label_row in zip(predictions, labels):
        seq_labels, seq_preds = [], []
        for pred_id, label_id in zip(pred_row, label_row):
            if label_id == -100:
                continue
            seq_labels.append(ID2LABEL[int(label_id)])
            seq_preds.append(ID2LABEL[int(pred_id)])
        true_labels.append(seq_labels)
        true_preds.append(seq_preds)

    return {
        "precision": precision_score(true_labels, true_preds),
        "recall":    recall_score(true_labels, true_preds),
        "f1":        f1_score(true_labels, true_preds),
    }


def train(config_path: str = "configs/deberta.yaml") -> None:
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).parents[3] / config_path

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))

    base = Path(__file__).parents[3]
    data_dir  = base / cfg["data_dir"]
    output_dir = base / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = cfg["model_name"]
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading datasets …")
    train_records = load_jsonl(data_dir / "train.jsonl")
    val_records   = load_jsonl(data_dir / "val.jsonl")
    print(f"  train={len(train_records)}  val={len(val_records)}")

    train_dataset = PIIDataset(train_records, tokenizer, LABEL2ID, cfg["max_length"])
    val_dataset   = PIIDataset(val_records,   tokenizer, LABEL2ID, cfg["max_length"])

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(UNIFIED_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"] * 2,
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup_ratio"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=cfg.get("fp16", True),
        logging_dir=str(output_dir / "logs"),
        logging_steps=50,
        report_to="none",
        seed=cfg.get("seed", 42),
    )

    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.get("patience", 3))],
    )

    print("Starting training …")
    trainer.train()

    best_model_dir = output_dir / "best_model"
    print(f"Saving best model → {best_model_dir}")
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))
    save_json(
        {"label2id": LABEL2ID, "id2label": {str(k): v for k, v in ID2LABEL.items()}},
        best_model_dir / "label_config.json",
    )
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/deberta.yaml")
    args = parser.parse_args()
    train(args.config)

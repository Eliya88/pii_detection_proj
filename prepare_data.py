#!/usr/bin/env python3
"""
Step 1 – Data Preparation
Loads both raw datasets, unifies labels, splits 80/10/10, and saves JSONL.

Usage:
    python prepare_data.py
    python prepare_data.py --kaggle path/to/kaggle.csv --ai4privacy path/to/pii43k.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.loaders import load_kaggle_dataset, load_ai4privacy_dataset
from src.data.splitter import stratified_split
from src.utils.io import save_jsonl

# Default paths (relative to this script)
DEFAULT_KAGGLE     = "pii_dataset.csv/pii_dataset.csv"   # nested as shipped
DEFAULT_AI4PRIVACY = "PII43k.csv"


def main(kaggle_path: str, ai4privacy_path: str) -> None:
    base = Path(__file__).parent

    kaggle_file     = base / kaggle_path
    ai4privacy_file = base / ai4privacy_path

    print(f"Loading Kaggle dataset from: {kaggle_file}")
    kaggle_records = load_kaggle_dataset(kaggle_file)
    print(f"  → {len(kaggle_records)} records")

    print(f"Loading ai4privacy dataset from: {ai4privacy_file}")
    ai4privacy_records = load_ai4privacy_dataset(ai4privacy_file)
    print(f"  → {len(ai4privacy_records)} records")

    all_records = kaggle_records + ai4privacy_records
    print(f"Total: {len(all_records)} records combined")

    train, val, test = stratified_split(all_records, train_ratio=0.80, val_ratio=0.10, seed=42)

    out_dir = base / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_jsonl(train, out_dir / "train.jsonl")
    save_jsonl(val,   out_dir / "val.jsonl")
    save_jsonl(test,  out_dir / "test.jsonl")

    print(f"\nSaved to {out_dir}/")
    print(f"  train.jsonl  – {len(train)} records")
    print(f"  val.jsonl    – {len(val)} records")
    print(f"  test.jsonl   – {len(test)} records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle",     default=DEFAULT_KAGGLE)
    parser.add_argument("--ai4privacy", default=DEFAULT_AI4PRIVACY)
    args = parser.parse_args()
    main(args.kaggle, args.ai4privacy)

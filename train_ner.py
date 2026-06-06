#!/usr/bin/env python3
"""
Step 2 – NER Model Training  (DeBERTa-v3-base)
Fine-tunes on train.jsonl, evaluates on val.jsonl, saves best checkpoint.

Usage:
    python train_ner.py
    python train_ner.py --config configs/deberta.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.models.ner_deberta.train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/deberta.yaml")
    args = parser.parse_args()
    train(args.config)

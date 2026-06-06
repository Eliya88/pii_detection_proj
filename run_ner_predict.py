#!/usr/bin/env python3
"""
Step 3c – NER Inference on Test Set
Runs the fine-tuned DeBERTa model on the held-out test set.

Usage:
    python run_ner_predict.py
    python run_ner_predict.py --model outputs/checkpoints/deberta/best_model
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.models.ner_deberta.predict import predict_and_save

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",   default="data/processed/test.jsonl")
    parser.add_argument("--model",  default="outputs/checkpoints/deberta/best_model")
    parser.add_argument("--output", default="outputs/results/ner_results.json")
    parser.add_argument("--batch",  type=int, default=32)
    args = parser.parse_args()
    predict_and_save(args.test, args.model, args.output, batch_size=args.batch)

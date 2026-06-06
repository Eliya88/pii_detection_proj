#!/usr/bin/env python3
"""
Step 3a – Presidio REGEX Pipeline
Runs Presidio Analyzer on the test set and evaluates with seqeval.

Usage:
    python run_regex.py
    python run_regex.py --test data/processed/test.jsonl --output outputs/results/regex_results.json
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.models.regex_presidio.predictor import run_presidio_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",      default="data/processed/test.jsonl")
    parser.add_argument("--output",    default="outputs/results/regex_results.json")
    parser.add_argument("--threshold", type=float, default=0.35)
    args = parser.parse_args()
    run_presidio_pipeline(args.test, args.output, args.threshold)

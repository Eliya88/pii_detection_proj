#!/usr/bin/env python3
"""
Step 3b – Gemini LLM Pipeline
Runs zero-shot / few-shot Gemini 1.5 Flash on a sampled test subset.

Usage:
    python run_llm.py
    python run_llm.py --config configs/llm.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.models.llm_gemini.predictor import run_llm_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/llm.yaml")
    args = parser.parse_args()
    run_llm_pipeline(args.config)

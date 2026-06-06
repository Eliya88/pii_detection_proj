#!/usr/bin/env python3
"""
Step 4 – Final Comparison
Loads all saved results and prints a unified metrics table.

Usage:
    python evaluate_all.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.evaluation.compare import run_comparison

if __name__ == "__main__":
    run_comparison(Path(__file__).parent)

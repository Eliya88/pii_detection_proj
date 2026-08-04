"""Gemini-based PII detection (Model 3 – LLM prompting)."""
from __future__ import annotations
import json
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from google import genai
from google.genai import types

from src.models.llm_gemini.prompts import SYSTEM_INSTRUCTION, build_prompt
from src.utils.io import (
    load_jsonl,
    save_json,
    tokens_to_text,
    tokens_to_text_with_offsets,
    char_spans_to_bio,
)
from src.evaluation.metrics import compute_metrics_from_sequences


def _extract_json(text: str) -> list:
    """Parse JSON array from model response, stripping markdown code fences."""
    text = text.strip()
    # Remove ```json ... ``` or ``` ... ``` fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Try parsing the whole response
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, list) else []
    except json.JSONDecodeError:
        pass

    # Fall back: find the first [...] block
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return []


def _validate_spans(spans: list, text: str) -> list:
    """Drop spans with unknown types or missing text field."""
    valid_types = {"PERSON", "EMAIL", "PHONE", "ADDRESS", "URL", "ID", "USERNAME"}
    clean = []
    for s in spans:
        if not isinstance(s, dict):
            continue
        if s.get("type") not in valid_types:
            continue
        if not s.get("text"):
            continue
        clean.append(s)
    return clean


def _fix_span_offsets(spans: list, text: str) -> list:
    """Replace Gemini's character offsets with real positions found by string search.

    Gemini knows *what* the PII text is but miscounts character positions in long
    documents. We use the entity text as a search key and pick the occurrence
    closest to Gemini's claimed start (so we handle repeated names correctly).
    """
    fixed = []
    for span in spans:
        entity_text = span["text"]
        claimed_start = span.get("start", 0) or 0

        # Find all exact occurrences
        occurrences = []
        pos = 0
        while True:
            idx = text.find(entity_text, pos)
            if idx == -1:
                break
            occurrences.append(idx)
            pos = idx + 1

        # Fall back to case-insensitive search
        if not occurrences:
            tl, el = text.lower(), entity_text.lower()
            pos = 0
            while True:
                idx = tl.find(el, pos)
                if idx == -1:
                    break
                occurrences.append(idx)
                pos = idx + 1

        if not occurrences:
            continue  # entity text genuinely not in source — skip

        # Pick occurrence nearest to Gemini's claimed position
        best = min(occurrences, key=lambda x: abs(x - claimed_start))
        fixed.append({**span, "start": best, "end": best + len(entity_text)})

    return fixed


def predict_record(text: str, tokens: list[str], client, few_shot_k: int,
                   trailing_ws=None, model_name: str = "gemini-1.5-flash",
                   gen_config=None) -> list[str]:
    prompt = build_prompt(text, few_shot_k)
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=gen_config,
        )
        raw = response.text
    except Exception as e:
        print(f"[Gemini] API error: {e}")
        return ["O"] * len(tokens)

    spans = _extract_json(raw)
    spans = _validate_spans(spans, text)
    spans = _fix_span_offsets(spans, text)  # anchor offsets to real text positions

    _, char_starts = tokens_to_text_with_offsets(tokens, trailing_ws)
    return char_spans_to_bio(tokens, char_starts, spans)


def run_llm_pipeline(config_path: str = "configs/llm.yaml") -> dict:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not set. Create a .env file with GEMINI_API_KEY=your_key"
        )

    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).parents[3] / config_path
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    base = Path(__file__).parents[3]
    data_dir   = base / cfg["data_dir"]
    output_dir = base / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    client = genai.Client(api_key=api_key)
    model_name = cfg["model_name"]
    gen_config = types.GenerateContentConfig(
        temperature=cfg.get("temperature", 0.0),
        max_output_tokens=cfg.get("max_output_tokens", 1024),
        system_instruction=SYSTEM_INSTRUCTION,
    )

    records = load_jsonl(data_dir / "test.jsonl")

    sample_size = cfg.get("sample_size", 500)
    rng = random.Random(cfg.get("seed", 42))
    if sample_size < len(records):
        records = rng.sample(records, sample_size)
        print(f"Sampled {sample_size} records from test set (cost control).")
    else:
        print(f"Running on all {len(records)} test records.")

    few_shot_k = cfg.get("few_shot_k", 5)
    preds: list[list[str]] = []
    total_tokens_in = 0
    total_tokens_out = 0
    t0 = time.perf_counter()

    for rec in tqdm(records, desc=f"Gemini ({model_name})"):
        tokens = rec["tokens"]
        trailing_ws = rec.get("trailing_whitespace")
        text = tokens_to_text(tokens, trailing_ws)

        bio = predict_record(text, tokens, client, few_shot_k, trailing_ws, model_name, gen_config)
        preds.append(bio)
        time.sleep(0.05)   # gentle rate-limit buffer

    elapsed = time.perf_counter() - t0
    latency_ms = (elapsed / len(records)) * 1000

    true_labels = [r["labels"] for r in records]
    metrics = compute_metrics_from_sequences(true_labels, preds)
    metrics["latency_ms_per_record"] = round(latency_ms, 3)
    metrics["records_evaluated"] = len(records)

    output_path = output_dir / "llm_results.json"
    result = {
        "metrics": metrics,
        "predictions": preds,
        "model": cfg["model_name"],
        "few_shot_k": few_shot_k,
    }
    save_json(result, output_path)
    print(
        f"Gemini → F1={metrics['f1']:.4f}  P={metrics['precision']:.4f}"
        f"  R={metrics['recall']:.4f}  latency={latency_ms:.1f} ms/rec"
    )
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/llm.yaml")
    args = parser.parse_args()
    run_llm_pipeline(args.config)

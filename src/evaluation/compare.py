"""Load all model results and produce a unified comparison report."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.io import load_json, save_json

MODEL_FILES = {
    "Presidio (REGEX)":     "outputs/results/regex_results.json",
    "DeBERTa-v3 (NER)":     "outputs/results/ner_results.json",
    "Gemini 2.5 Flash (LLM)":"outputs/results/llm_results.json",
}

ENTITY_TYPES = ["PERSON", "EMAIL", "PHONE", "ADDRESS", "URL", "ID", "USERNAME"]


def _row(name: str, result: dict) -> dict:
    m = result["metrics"]
    row = {
        "model": name,
        "precision": m.get("precision", 0),
        "recall":    m.get("recall", 0),
        "f1":        m.get("f1", 0),
        "latency_ms": m.get("latency_ms_per_record", "-"),
    }
    for ent in ENTITY_TYPES:
        per = m.get("per_entity", {}).get(ent, {})
        row[f"{ent}_f1"] = per.get("f1", 0)
    return row


def build_comparison(base_dir: Path | None = None) -> list[dict]:
    if base_dir is None:
        base_dir = Path(__file__).parents[2]

    rows = []
    for name, rel_path in MODEL_FILES.items():
        path = base_dir / rel_path
        if not path.exists():
            print(f"[compare] Missing result file: {path}  — skipping {name}")
            continue
        result = load_json(path)
        rows.append(_row(name, result))

    return rows


def print_table(rows: list[dict]) -> None:
    header = f"{'Model':<30} {'P':>6} {'R':>6} {'F1':>6} {'Latency':>12}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        lat = f"{r['latency_ms']:.1f} ms" if isinstance(r["latency_ms"], float) else "-"
        print(
            f"{r['model']:<30} {r['precision']:>6.4f} {r['recall']:>6.4f}"
            f" {r['f1']:>6.4f} {lat:>12}"
        )
    print(sep)

    print("\nPer-entity F1:")
    ent_header = f"{'Model':<30} " + " ".join(f"{e[:7]:>8}" for e in ENTITY_TYPES)
    print(ent_header)
    print("-" * len(ent_header))
    for r in rows:
        ent_vals = " ".join(f"{r.get(f'{e}_f1', 0):>8.4f}" for e in ENTITY_TYPES)
        print(f"{r['model']:<30} {ent_vals}")


def run_comparison(base_dir: Path | None = None) -> None:
    base_dir = base_dir or Path(__file__).parents[2]
    rows = build_comparison(base_dir)
    if not rows:
        print("No result files found. Run the model pipelines first.")
        return

    print_table(rows)

    out = base_dir / "outputs" / "results" / "comparison.json"
    save_json({"models": rows}, out)
    print(f"\nComparison saved → {out}")


if __name__ == "__main__":
    run_comparison()

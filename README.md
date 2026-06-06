# PII Detection in Text Using Multiple Approaches

**Group 15** — Eliya Ohayon, Ido Aloya, Omer Dweck  
Ben-Gurion University of the Negev — Data Science & Big Data in Industry

---

## Overview

This project compares three approaches for detecting Personally Identifiable Information (PII) in text, following the CRISP-DM methodology. We evaluate a rule-based system, a fine-tuned transformer, and a large language model on a unified benchmark.

---

## Models

| Model | Type | F1 | Latency |
|---|---|---|---|
| Microsoft Presidio | Rule-based (REGEX + NLP) | 42.1% | 14 ms/rec |
| DeBERTa-v3-base | Fine-tuned NER | **99.0%** | — |
| Gemini 2.5 Flash | LLM (few-shot prompting) | 17.6% | 2196 ms/rec |

---

## Unified Label Schema

7 entity types mapped from both datasets:

`PERSON` · `EMAIL` · `PHONE` · `ADDRESS` · `URL` · `ID` · `USERNAME`

---

## Datasets

| Dataset | Records | Source |
|---|---|---|
| Kaggle PII Data Detection | 22,000 | Student essays, word-level BIO |
| ai4privacy PII Masking (PII43k) | 43,000 | General text, WordPiece BIO |

Combined and split 80/10/10 (train/val/test) with stratification by PII presence.  
**Final split:** train=37,661 · val=4,707 · test=4,710

---

## Project Structure

```
pii_detection/
├── src/
│   ├── data/               # Loaders, label mapping, splitting
│   ├── models/
│   │   ├── regex_presidio/ # Model 1 — Presidio pipeline
│   │   ├── ner_deberta/    # Model 2 — DeBERTa fine-tuning & inference
│   │   └── llm_gemini/     # Model 3 — Gemini prompting
│   ├── evaluation/         # seqeval metrics, comparison table
│   └── utils/              # I/O helpers
├── notebooks/              # 6 presentation notebooks
│   ├── 01_EDA.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_regex_pipeline.ipynb
│   ├── 04_ner_pipeline.ipynb
│   ├── 05_llm_pipeline.ipynb
│   └── 06_comparison_results.ipynb
├── configs/                # YAML configs for each model
├── outputs/results/        # Pre-computed evaluation results (JSON)
├── prepare_data.py         # Step 1 — data preparation
├── train_ner.py            # Step 2 — DeBERTa training (GPU required)
├── run_regex.py            # Step 3a — Presidio evaluation
├── run_ner_predict.py      # Step 3b — DeBERTa inference
├── run_llm.py              # Step 3c — Gemini evaluation
├── evaluate_all.py         # Step 4 — comparison table
├── train_job.sh            # SLURM job script (BGU cluster)
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

For Gemini (Model 3), create a `.env` file:
```
GEMINI_API_KEY=your_key_here
```

---

## Reproducing Results

```bash
python prepare_data.py       # Step 1 — data prep (run once)
python train_ner.py          # Step 2 — ~3h on RTX 3090
python run_regex.py          # Step 3a — seconds
python run_ner_predict.py    # Step 3b — minutes (GPU recommended)
python run_llm.py            # Step 3c — ~18 min, ~$0.04 API cost
python evaluate_all.py       # Step 4 — prints comparison table
```

DeBERTa training was performed on the BGU HPC cluster (RTX 3090 24GB, 10 epochs, ~2.85 hours).

---

## Key Findings

- **Fine-tuned DeBERTa** achieves near-perfect F1 (99%) across all entity types, demonstrating the power of supervised NER on in-domain data.
- **Presidio** performs well on structured PII (PHONE: 72%, ID: 50%) but struggles with context-dependent entities like ADDRESS.
- **Gemini** shows lower strict F1 due to span boundary mismatches under seqeval's exact-match criterion, not because it fails to understand PII conceptually.
- **Speed vs. accuracy tradeoff:** Presidio is fastest (14 ms), DeBERTa is most accurate, Gemini is slowest (2.2 s) with highest per-call cost.

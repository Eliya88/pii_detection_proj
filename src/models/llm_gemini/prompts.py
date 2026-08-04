"""Prompt templates for Gemini PII extraction."""
from __future__ import annotations

ENTITY_DESCRIPTIONS = """PERSON      – full or partial person name (first name, last name, full name)
EMAIL       – email address
PHONE       – phone or fax number
ADDRESS     – street address, city, state, ZIP / postal code, country
URL         – web URL, domain name, IP address
ID          – government ID, SSN, passport, driver's licence, credit-card number, IBAN
USERNAME    – social-media handle, login name, @mention"""

SYSTEM_INSTRUCTION = (
    "You are a precise PII extraction engine. "
    "Return ONLY a valid JSON array — no prose, no markdown fences."
)

_ZERO_SHOT_TEMPLATE = """\
Identify all PII spans in the text below and return them as a JSON array.
Each element must have exactly these keys:
  "text"  – the exact substring as it appears in the input
  "type"  – one of: PERSON, EMAIL, PHONE, ADDRESS, URL, ID, USERNAME
  "start" – character offset of the first character of the span
  "end"   – character offset one past the last character of the span

Entity type guide:
{entity_descriptions}

If no PII is found, return an empty array: []

Text:
{text}
"""

_FEW_SHOT_EXAMPLE_TEMPLATE = """\
Text:
{text}

Output:
{output}
"""

_FEW_SHOT_TEMPLATE = """\
Identify all PII spans in the text below and return them as a JSON array.
Each element must have exactly these keys:
  "text"  – the exact substring as it appears in the input
  "type"  – one of: PERSON, EMAIL, PHONE, ADDRESS, URL, ID, USERNAME
  "start" – character offset of the first character of the span
  "end"   – character offset one past the last character of the span

Entity type guide:
{entity_descriptions}

If no PII is found, return an empty array: []

--- Examples ---
{examples}
--- End of examples ---

Text:
{text}
"""

# Few-shot examples sourced from training data (train.jsonl), covering all 7 entity types.
# Each example is a real record from the dataset with verified character offsets.
FEW_SHOT_EXAMPLES = [
    # 1. PERSON + EMAIL + ADDRESS (from ai4privacy training data)
    {
        "text": "draft an email to pamela sanford at virginia23 @ gmail . com explaining the new overtime regulations in maine .",
        "entities": [
            {"text": "pamela sanford", "type": "PERSON", "start": 18, "end": 32},
            {"text": "virginia23 @ gmail . com", "type": "EMAIL", "start": 36, "end": 60},
            {"text": "maine", "type": "ADDRESS", "start": 104, "end": 109},
        ],
    },
    # 2. PERSON + USERNAME + ID (from ai4privacy training data)
    {
        "text": "14 . provide a summary of the regulations governing bethany sauer ' s auto loan account and 02051137 .",
        "entities": [
            {"text": "bethany sauer", "type": "PERSON", "start": 52, "end": 65},
            {"text": "auto loan account", "type": "USERNAME", "start": 70, "end": 87},
            {"text": "02051137", "type": "ID", "start": 92, "end": 100},
        ],
    },
    # 3. PERSON + ADDRESS + ID (from ai4privacy training data)
    {
        "text": "create a sample living trust document for wisoky group that includes their real estate property at 415 roosevelt manors and investments with account number 90774169 .",
        "entities": [
            {"text": "wisoky group", "type": "PERSON", "start": 42, "end": 54},
            {"text": "415 roosevelt manors", "type": "ADDRESS", "start": 99, "end": 119},
            {"text": "90774169", "type": "ID", "start": 156, "end": 164},
        ],
    },
    # 4. PERSON + EMAIL + ADDRESS (from ai4privacy training data)
    {
        "text": "hey , can you send a list of the top 5 legal tech startups in iowa to janet aufderhar at genevieve88 @ gmail . com ?",
        "entities": [
            {"text": "iowa", "type": "ADDRESS", "start": 62, "end": 66},
            {"text": "janet aufderhar", "type": "PERSON", "start": 70, "end": 85},
            {"text": "genevieve88 @ gmail . com", "type": "EMAIL", "start": 89, "end": 114},
        ],
    },
    # 5. EMAIL + URL (from ai4privacy training data)
    {
        "text": "what should i consider when collecting weston _ kunde40 @ yahoo . com and 5d8f : 2ded : f141 : 6103 : 2ab8 : cd2f : 623e : b9d4 addresses to maintain user privacy ?",
        "entities": [
            {"text": "weston _ kunde40 @ yahoo . com", "type": "EMAIL", "start": 39, "end": 69},
            {"text": "5d8f : 2ded : f141 : 6103 : 2ab8 : cd2f : 623e : b9d4", "type": "URL", "start": 74, "end": 127},
        ],
    },
    # 6. PERSON + EMAIL + ID (from ai4privacy training data)
    {
        "text": "draft a letter to hagenes llc at brennan _ ernser38 @ hotmail . com requesting documentation related to their financial accounts , including 99171989 .",
        "entities": [
            {"text": "hagenes llc", "type": "PERSON", "start": 18, "end": 29},
            {"text": "brennan _ ernser38 @ hotmail . com", "type": "EMAIL", "start": 33, "end": 67},
            {"text": "99171989", "type": "ID", "start": 141, "end": 149},
        ],
    },
    # 7. PERSON + ADDRESS + ID (from ai4privacy training data)
    {
        "text": "write a legal document for orn , morar and halvorson outlining the division of marital property , including the assets in 68973105 and the property at 974 maggio road .",
        "entities": [
            {"text": "orn , morar and halvorson", "type": "PERSON", "start": 27, "end": 52},
            {"text": "68973105", "type": "ID", "start": 122, "end": 130},
            {"text": "974 maggio road", "type": "ADDRESS", "start": 151, "end": 166},
        ],
    },
    # 8. PERSON + EMAIL + ID (from ai4privacy training data)
    {
        "text": "write a guide for dach llc on how to conduct a privacy impact assessment when processing 38318100 and dell87 @ hotmail . com data .",
        "entities": [
            {"text": "dach llc", "type": "PERSON", "start": 18, "end": 26},
            {"text": "38318100", "type": "ID", "start": 89, "end": 97},
            {"text": "dell87 @ hotmail . com", "type": "EMAIL", "start": 102, "end": 124},
        ],
    },
    # 9. PERSON + EMAIL + PHONE + ADDRESS (from Kaggle training data — extracted segment)
    {
        "text": "Personal Information: Name: Masako Novikov, Email: masako_novikov@yahoo.org, Phone: +91-84874 79121, Address: 16 Clark Street, Hobby: Video Production",
        "entities": [
            {"text": "Masako Novikov", "type": "PERSON", "start": 28, "end": 42},
            {"text": "masako_novikov@yahoo.org", "type": "EMAIL", "start": 51, "end": 74},
            {"text": "+91-84874 79121", "type": "PHONE", "start": 83, "end": 98},
            {"text": "16 Clark Street", "type": "ADDRESS", "start": 109, "end": 124},
        ],
    },
    # 10. PERSON + ADDRESS + EMAIL (from ai4privacy training data)
    {
        "text": "please send dora hegmann a list of top 10 child psychologists in fort darrickfield to caitlyn _ carter @ gmail . com .",
        "entities": [
            {"text": "dora hegmann", "type": "PERSON", "start": 12, "end": 24},
            {"text": "fort darrickfield", "type": "ADDRESS", "start": 64, "end": 81},
            {"text": "caitlyn _ carter @ gmail . com", "type": "EMAIL", "start": 85, "end": 115},
        ],
    },
]


def _format_example(ex: dict) -> str:
    import json
    return _FEW_SHOT_EXAMPLE_TEMPLATE.format(
        text=ex["text"],
        output=json.dumps(ex["entities"], indent=2),
    )


def build_prompt(text: str, few_shot_k: int = 0) -> str:
    if few_shot_k == 0:
        return _ZERO_SHOT_TEMPLATE.format(
            entity_descriptions=ENTITY_DESCRIPTIONS,
            text=text,
        )

    examples_text = "\n".join(
        _format_example(ex) for ex in FEW_SHOT_EXAMPLES[:few_shot_k]
    )
    return _FEW_SHOT_TEMPLATE.format(
        entity_descriptions=ENTITY_DESCRIPTIONS,
        examples=examples_text,
        text=text,
    )

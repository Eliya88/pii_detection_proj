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

# Hard-coded few-shot examples (diverse entity coverage)
FEW_SHOT_EXAMPLES = [
    {
        "text": "My name is Sarah Connor and you can reach me at sarah.c@example.com or (555) 867-5309.",
        "entities": [
            {"text": "Sarah Connor", "type": "PERSON", "start": 11, "end": 23},
            {"text": "sarah.c@example.com", "type": "EMAIL", "start": 47, "end": 66},
            {"text": "(555) 867-5309", "type": "PHONE", "start": 70, "end": 84},
        ],
    },
    {
        "text": "Visit our office at 42 Wallaby Way, Sydney or go to https://ourcompany.io for details.",
        "entities": [
            {"text": "42 Wallaby Way, Sydney", "type": "ADDRESS", "start": 20, "end": 42},
            {"text": "https://ourcompany.io", "type": "URL", "start": 50, "end": 71},
        ],
    },
    {
        "text": "Account holder: John Doe (SSN 123-45-6789). Username: @johnd_official.",
        "entities": [
            {"text": "John Doe", "type": "PERSON", "start": 17, "end": 25},
            {"text": "123-45-6789", "type": "ID", "start": 31, "end": 42},
            {"text": "@johnd_official", "type": "USERNAME", "start": 54, "end": 69},
        ],
    },
    {
        "text": "Send the invoice to billing@acme.org — IBAN: GB29 NWBK 6016 1331 9268 19.",
        "entities": [
            {"text": "billing@acme.org", "type": "EMAIL", "start": 20, "end": 36},
            {"text": "GB29 NWBK 6016 1331 9268 19", "type": "ID", "start": 45, "end": 72},
        ],
    },
    {
        "text": "Dr. Emily Zhang lives at 501 Oak Street, Portland, OR 97201. Her IP is 192.168.1.100.",
        "entities": [
            {"text": "Emily Zhang", "type": "PERSON", "start": 4, "end": 15},
            {"text": "501 Oak Street, Portland, OR 97201", "type": "ADDRESS", "start": 25, "end": 59},
            {"text": "192.168.1.100", "type": "URL", "start": 71, "end": 84},
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

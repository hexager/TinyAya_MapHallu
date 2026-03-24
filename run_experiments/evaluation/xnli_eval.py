"""
evaluation/xnli_eval.py — Gold-label comparison for XNLI NLI task.

Parses the model's JSON response to extract the predicted label,
then compares to the XNLI gold label.
"""

import json


def parse_label(response_text: str) -> str | None:
    """
    Extract the NLI label from the model's JSON response.

    Expected format: {"label": "entailment"}
    Returns normalised label string, or None on parse failure.
    """
    try:
        obj = json.loads(response_text)
        label = obj.get("label", "").strip().lower()
        return label if label else None
    except (json.JSONDecodeError, AttributeError):
        return None


def is_correct(response_text: str, gold_label: str) -> bool:
    """True if the parsed label matches the gold label."""
    parsed = parse_label(response_text)
    if parsed is None:
        return False
    return parsed == gold_label.strip().lower()

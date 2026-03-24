"""
evaluation/mkqa_eval.py — Gold-answer substring match for MKQA QA task.

Adapted from hallucination-rate/metrics.py.
"""

import json
import re
from typing import List


def _normalize(s: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    if not s or not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s.lower().strip())


def get_acceptable_answer_strings(answers: List[dict]) -> List[str]:
    """Extract all acceptable normalised answer strings from MKQA answer list."""
    out = []
    for a in answers or []:
        if not isinstance(a, dict):
            continue
        t = a.get("text")
        if t and isinstance(t, str) and t.strip():
            out.append(_normalize(t))
        for alias in (a.get("aliases") or []):
            if alias and isinstance(alias, str) and alias.strip():
                out.append(_normalize(alias))
    return list(dict.fromkeys(out))


def parse_answer(response_text: str) -> str:
    """
    Extract answer text from the model's JSON response.

    Expected format: {"answer": "Paris"}
    Falls back to raw text if JSON parsing fails.
    """
    try:
        obj = json.loads(response_text)
        return obj.get("answer", response_text) or response_text
    except (json.JSONDecodeError, AttributeError):
        return response_text


def is_correct(response_text: str, gold_answers: List[dict]) -> bool:
    """
    True if the model response contains at least one gold answer substring.

    Parses JSON answer field first, then checks containment.
    """
    gold_strings = get_acceptable_answer_strings(gold_answers)
    if not gold_strings:
        return False

    answer_text = parse_answer(response_text)
    resp_norm = _normalize(answer_text)
    if not resp_norm:
        return False

    return any(g in resp_norm for g in gold_strings)

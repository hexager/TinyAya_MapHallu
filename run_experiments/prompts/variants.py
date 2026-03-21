"""
prompts/variants.py — Rule-based prompt variant generator for PSS experiments.

Adapted from Prompt Sensitivity Score/prompt_variants.py.
Generates 4 deterministic variants per base prompt (no model calls).
"""

import re
from typing import Dict, List

_PARAPHRASE_PREFIXES = [
    "Could you tell me",
    "I'd like to know",
    "Please explain",
    "Can you describe",
]

_INSTRUCTION_WRAPPERS = [
    "Please answer the following question as accurately as possible: {prompt}",
    "Answer this question in detail: {prompt}",
]

_CONTEXT_ADDITIONS = [
    "{prompt} Please use simple language.",
    "{prompt} Be concise and factual.",
]

_STRIP_RE = re.compile(
    r"^(what is|what are|what was|what were"
    r"|who is|who was|who were"
    r"|where is|where was|where are"
    r"|when did|when was|when were"
    r"|how does|how do|how did|how is"
    r"|can you tell me|please tell me|tell me"
    r"|could you tell me|i'd like to know"
    r"|please explain|can you describe)\s+",
    re.IGNORECASE,
)


def _paraphrase(prompt: str) -> str:
    stripped = _STRIP_RE.sub("", prompt).rstrip("?").strip()
    return f"{_PARAPHRASE_PREFIXES[0]} {stripped}?"


def _instruction_format(prompt: str) -> str:
    return _INSTRUCTION_WRAPPERS[0].format(prompt=prompt)


def _context_addition(prompt: str) -> str:
    return _CONTEXT_ADDITIONS[0].format(prompt=prompt)


def _shorten(prompt: str) -> str:
    shortened = _STRIP_RE.sub("", prompt).strip()
    if not shortened:
        return prompt
    return shortened[0].upper() + shortened[1:]


def generate_variants(base_prompt: str) -> List[Dict[str, str]]:
    """
    Return 4 variant dicts for *base_prompt*.

    Each dict: {"variant_type": str, "variant_prompt": str}
    """
    return [
        {"variant_type": "paraphrase", "variant_prompt": _paraphrase(base_prompt)},
        {"variant_type": "instruction", "variant_prompt": _instruction_format(base_prompt)},
        {"variant_type": "context", "variant_prompt": _context_addition(base_prompt)},
        {"variant_type": "short", "variant_prompt": _shorten(base_prompt)},
    ]

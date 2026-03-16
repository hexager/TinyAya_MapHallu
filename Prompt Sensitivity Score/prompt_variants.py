"""
prompt_variants.py — Rule-based generator for four task-preserving prompt variants.

Variant types
-------------
1. paraphrase   – Rephrase the question with a different syntactic opener.
2. instruction  – Wrap the prompt in an explicit instruction frame.
3. context      – Append a short framing phrase after the prompt.
4. short        – Strip filler openers to produce a compressed version.

All transformations are deterministic (no extra model calls), so they add
zero latency and are fully reproducible across runs.

Example
-------
    base = "What is the boiling point of water?"
    variants = generate_variants(base)
    # → paraphrase : "Could you tell me the boiling point of water?"
    # → instruction: "Please answer the following question as accurately as
    #                 possible: What is the boiling point of water?"
    # → context    : "What is the boiling point of water? Please use simple
    #                 language."
    # → short      : "Boiling point of water?"
"""

import re
from typing import Dict, List

# ---------------------------------------------------------------------------
# Template banks  (index 0 is always used for determinism)
# ---------------------------------------------------------------------------

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

# Wh-question / filler openers to strip when shortening.
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


# ---------------------------------------------------------------------------
# Individual variant functions
# ---------------------------------------------------------------------------

def _paraphrase(prompt: str) -> str:
    """
    Rephrase using a declarative opener instead of the original wh-structure.

    Example:
        "What is photosynthesis?"  →  "Could you tell me photosynthesis?"
    """
    stripped = _STRIP_RE.sub("", prompt).rstrip("?").strip()
    return f"{_PARAPHRASE_PREFIXES[0]} {stripped}?"


def _instruction_format(prompt: str) -> str:
    """
    Wrap the prompt in an explicit instruction frame.

    Example:
        "What is photosynthesis?"
        →  "Please answer the following question as accurately as possible:
            What is photosynthesis?"
    """
    return _INSTRUCTION_WRAPPERS[0].format(prompt=prompt)


def _context_addition(prompt: str) -> str:
    """
    Append a short framing phrase.

    Example:
        "What is photosynthesis?"
        →  "What is photosynthesis? Please use simple language."
    """
    return _CONTEXT_ADDITIONS[0].format(prompt=prompt)


def _shorten(prompt: str) -> str:
    """
    Produce a compressed version by stripping filler openers.

    Example:
        "What is the boiling point of water?"  →  "Boiling point of water?"
        "Could you please explain photosynthesis?"  →  "Photosynthesis?"
    """
    shortened = _STRIP_RE.sub("", prompt).strip()
    if not shortened:
        return prompt  # fallback: return original if stripping removed everything
    # Capitalise the first character and keep the rest as-is.
    return shortened[0].upper() + shortened[1:]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_variants(base_prompt: str) -> List[Dict[str, str]]:
    """
    Return a list of four variant dicts for ``base_prompt``.

    Each dict has:
        variant_type  : str  — "paraphrase" | "instruction" | "context" | "short"
        variant_prompt: str  — the modified prompt text

    Parameters
    ----------
    base_prompt : str
        The (possibly language-adapted) base prompt text.

    Returns
    -------
    list of dict  (always length 4, in the order listed above)
    """
    return [
        {
            "variant_type":   "paraphrase",
            "variant_prompt": _paraphrase(base_prompt),
        },
        {
            "variant_type":   "instruction",
            "variant_prompt": _instruction_format(base_prompt),
        },
        {
            "variant_type":   "context",
            "variant_prompt": _context_addition(base_prompt),
        },
        {
            "variant_type":   "short",
            "variant_prompt": _shorten(base_prompt),
        },
    ]

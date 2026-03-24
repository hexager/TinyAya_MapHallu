"""
data/load_xnli.py — Load XNLI dataset in standardized format.

Adapted from CMDR/CMDR.py load_multilingual_data().
Returns a flat list of sample dicts with consistent schema.
"""

import logging
import os

from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# XNLI integer label → string
_LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}

XNLI_LANGUAGES = {"ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"}


def load_xnli(languages: list[str], num_samples: int | None = 300, seed: int = 42) -> list[dict]:
    """
    Load aligned XNLI samples for each language.

    If num_samples is None, all samples are used.

    Returns list of dicts:
        {
            "sample_id": int,
            "language": str,
            "prompt_fields": {"premise": str, "hypothesis": str},
            "gold_label": str,       # "entailment" | "neutral" | "contradiction"
        }

    Samples are aligned by index across languages (XNLI guarantees this).
    """
    rows = []
    for lang in languages:
        if lang not in XNLI_LANGUAGES:
            logger.warning("Skipping language '%s' — not available in XNLI. Valid: %s", lang, sorted(XNLI_LANGUAGES))
            continue
        ds = load_dataset("xnli", lang, split="test", token=os.getenv("HF_TOKEN"))
        n = len(ds) if num_samples is None else min(num_samples, len(ds))
        for i in range(n):
            rows.append({
                "sample_id": i,
                "language": lang,
                "prompt_fields": {
                    "premise": ds[i]["premise"],
                    "hypothesis": ds[i]["hypothesis"],
                },
                "gold_label": _LABEL_MAP.get(ds[i]["label"], str(ds[i]["label"])),
            })
    return rows

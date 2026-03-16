"""
utils/entity_utils.py — Named-entity extraction helpers.

Public API
----------
extract_entities(text, language)  NER via spaCy (EN) or regex fallback.
primary_entity(entities)          First entity string lowercased, or None.

Notes
-----
• Arabic and Hindi use non-Latin scripts; the regex fallback returns an
  empty list for them, so entity_change_rate will be null for those languages.
• Install spaCy for improved English coverage:
      pip install spacy && python -m spacy download en_core_web_sm
"""

import logging
import re
import warnings
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# spaCy — optional; loaded once at import time.
# ---------------------------------------------------------------------------
_SPACY_NLP_EN    = None
_SPACY_AVAILABLE = False

try:
    import spacy as _spacy
    _SPACY_NLP_EN    = _spacy.load("en_core_web_sm")
    _SPACY_AVAILABLE = True
    logger.info("spaCy en_core_web_sm loaded.")
except Exception:
    warnings.warn(
        "spaCy 'en_core_web_sm' not found — using regex fallback for entity extraction.\n"
        "To enable spaCy: pip install spacy && python -m spacy download en_core_web_sm",
        UserWarning,
        stacklevel=1,
    )


def _spacy_entities(text: str) -> List[str]:
    """Extract named entities using spaCy (English pipeline)."""
    doc = _SPACY_NLP_EN(text)
    return [ent.text for ent in doc.ents]


def _regex_entities(text: str) -> List[str]:
    """
    Regex fallback: contiguous Title-Case tokens as proxy for named entities.

    Works for Latin-script languages (en, es, fr …).
    Returns an empty list for Arabic / Hindi (entity_change_rate → null).
    """
    candidates = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text)
    return [c for c in candidates if len(c) > 1]


def extract_entities(text: str, language: str = "en") -> Optional[List[str]]:
    """
    Extract named entities from ``text``.

    Strategy
    --------
    • English + spaCy available  → spaCy en_core_web_sm
    • All other cases            → Title-Case regex heuristic
    • Empty / blank text         → None  (logs a warning)

    Parameters
    ----------
    text     : str
    language : str  ISO 639-1 code

    Returns
    -------
    list of str, or None on failure
    """
    if not text or not text.strip():
        logger.warning("Empty text for entity extraction (lang=%s).", language)
        return None
    try:
        if _SPACY_AVAILABLE and language == "en":
            return _spacy_entities(text)
        return _regex_entities(text)
    except Exception as exc:
        logger.warning("Entity extraction error (lang=%s): %s", language, exc)
        return None


def primary_entity(entities: Optional[List[str]]) -> Optional[str]:
    """Return the first entity lowercased, or None."""
    if entities:
        return entities[0].lower().strip()
    return None

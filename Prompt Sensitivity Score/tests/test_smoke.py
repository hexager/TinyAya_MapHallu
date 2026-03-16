"""
tests/test_smoke.py — Minimal smoke tests for the PSS pipeline.

Tests
-----
1. JSON schema validation       — all required fields present in every record.
2. word-token Jaccard           — identical texts → 1.0; disjoint → 0.0.
3. avg_cosine_vs_base           — identical embeddings → 1.0; orthogonal → 0.0.
4. lexical_overlap_vs_base      — identical texts → 1.0.
5. response_length_variance     — same-length texts → 0.0; varying → > 0.
6. Entity extraction            — finds entity in clear English sentence.
7. Empty text entity extraction — returns None without crashing.

Run with pytest:
    python -m pytest tests/ -v

Or directly:
    python tests/test_smoke.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from analysis.utils import (
    avg_cosine_vs_base,
    extract_entities,
    jaccard,
    lexical_overlap_vs_base,
    primary_entity,
    response_length_variance,
    word_token_set,
)

# ---------------------------------------------------------------------------
# Required fields for every raw_outputs.jsonl record
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = {
    "run_id",
    "prompt_id",
    "language",
    "variant_type",
    "base_prompt",
    "variant_prompt",
    "model",
    "temperature",
    "response",
    "timestamp",
}


def _assert(condition: bool, msg: str) -> None:
    if not condition:
        raise AssertionError(msg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_json_schema() -> None:
    """Verify that a synthetic record satisfies the required-field contract."""
    record = {
        "run_id":         "test_run_001",
        "prompt_id":      0,
        "language":       "en",
        "variant_type":   "base",
        "base_prompt":    "What is the capital of France?",
        "variant_prompt": "What is the capital of France?",
        "model":          "c4ai-aya-expanse-8b",
        "temperature":    0,
        "response":       "The capital of France is Paris.",
        "response_tokens": 8,
        "timestamp":      "2024-01-01T00:00:00+00:00",
    }
    missing = REQUIRED_FIELDS - set(record.keys())
    _assert(not missing, f"Record missing fields: {missing}")
    print("PASS  test_json_schema")


def test_word_token_jaccard() -> None:
    """Jaccard of identical word sets == 1.0; disjoint == 0.0."""
    text = "the quick brown fox"
    _assert(
        jaccard(word_token_set(text), word_token_set(text)) == 1.0,
        "Expected 1.0 for identical word sets",
    )
    _assert(
        jaccard(word_token_set("alpha beta gamma"), word_token_set("delta epsilon zeta")) == 0.0,
        "Expected 0.0 for disjoint word sets",
    )
    print("PASS  test_word_token_jaccard")


def test_avg_cosine_vs_base() -> None:
    """Identical embeddings → 1.0; orthogonal → 0.0; single row → 1.0."""
    e_same = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    _assert(abs(avg_cosine_vs_base(e_same) - 1.0) < 1e-6,
            "Expected ~1.0 for identical embeddings")

    e_orth = np.array([[1.0, 0.0], [0.0, 1.0]])
    _assert(abs(avg_cosine_vs_base(e_orth) - 0.0) < 1e-6,
            "Expected ~0.0 for orthogonal embeddings")

    _assert(avg_cosine_vs_base(np.array([[0.6, 0.8]])) == 1.0,
            "Expected 1.0 for single-row input")
    print("PASS  test_avg_cosine_vs_base")


def test_lexical_overlap_vs_base() -> None:
    """Identical texts → 1.0; single text → 1.0."""
    texts = ["the cat sat on the mat"] * 3
    _assert(abs(lexical_overlap_vs_base(texts) - 1.0) < 1e-6,
            "Expected 1.0 for identical texts")
    _assert(lexical_overlap_vs_base(["only one text"]) == 1.0,
            "Expected 1.0 for single text (edge case)")
    print("PASS  test_lexical_overlap_vs_base")


def test_response_length_variance() -> None:
    """Same-length texts → 0.0; different lengths → > 0."""
    same   = ["hello world"] * 4          # all length 2
    varied = ["hi", "hello world", "the quick brown fox jumps"]
    _assert(response_length_variance(same)   == 0.0,
            "Expected 0.0 variance for equal-length responses")
    _assert(response_length_variance(varied)  > 0.0,
            "Expected > 0 variance for varying-length responses")
    _assert(response_length_variance(["only"]) == 0.0,
            "Expected 0.0 for single text")
    print("PASS  test_response_length_variance")


def test_entity_extraction() -> None:
    """Should find at least one entity in a clear English sentence."""
    ents = extract_entities("Paris is the capital of France.", "en")
    _assert(ents is not None, "Expected non-None entity list")
    _assert(len(ents) > 0, "Expected ≥1 entity (spaCy or regex fallback)")
    _assert(primary_entity(ents) is not None, "primary_entity returned None")
    print("PASS  test_entity_extraction")


def test_empty_text_entity() -> None:
    """Empty text → None, no crash."""
    _assert(extract_entities("", "en") is None,
            "Expected None for empty text")
    print("PASS  test_empty_text_entity")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_json_schema()
    test_word_token_jaccard()
    test_avg_cosine_vs_base()
    test_lexical_overlap_vs_base()
    test_response_length_variance()
    test_entity_extraction()
    test_empty_text_entity()
    print("\nAll smoke tests passed.")

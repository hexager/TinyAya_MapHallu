"""
utils/validate_dataset.py — Stage 0: Validate a raw_outputs dataset.

Checks the collected JSONL / JSON dataset before running any analysis:
  1. Required fields exist in every record.
  2. Correct number of variants per (prompt_id, language) group.
  3. No empty responses.
  4. Language values are from the expected set.

Prints clear warnings for any issues found and exits with code 1 if
critical problems are detected (missing fields, wrong variant counts).

Usage
-----
    python utils/validate_dataset.py
    python utils/validate_dataset.py --input data/raw_outputs.json
    python utils/validate_dataset.py --input data/raw_outputs.json --strict
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Expected schema
# ---------------------------------------------------------------------------
# New schema (src/collect_data.py)
REQUIRED_FIELDS = {
    "prompt_id", "prompt_text", "language", "variant_type",
    "model", "temperature", "response", "response_length",
}

# Legacy schema (root collect_data.py) — accepted as fallback
REQUIRED_FIELDS_LEGACY = {
    "run_id", "prompt_id", "language", "variant_type",
    "base_prompt", "variant_prompt", "model",
    "temperature", "response", "timestamp",
}

EXPECTED_VARIANTS = {"base", "paraphrase", "instruction", "context", "short"}
EXPECTED_LANGUAGES = {"en", "es", "hi", "ar"}


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------

def check_required_fields(records: list) -> int:
    """
    Return count of records missing required fields.

    Detects schema version automatically — accepts both new schema
    (prompt_text, response_length) and legacy schema (base_prompt, run_id).
    """
    if not records:
        return 0
    # Detect schema from first record.
    first = set(records[0].keys())
    schema = REQUIRED_FIELDS if "prompt_text" in first else REQUIRED_FIELDS_LEGACY
    logger.info("Detected schema: %s", "new" if schema is REQUIRED_FIELDS else "legacy")

    bad = 0
    for i, rec in enumerate(records):
        missing = schema - set(rec.keys())
        if missing:
            logger.warning("Record %d missing fields: %s", i, missing)
            bad += 1
    return bad


def check_variant_counts(records: list, expected_variants: set) -> int:
    """
    Verify each (prompt_id, language) group has exactly the expected variants.
    Returns number of groups with wrong variant sets.
    """
    groups: dict = defaultdict(set)
    for rec in records:
        key = (rec.get("prompt_id"), rec.get("language"))
        groups[key].add(rec.get("variant_type"))

    bad = 0
    for (pid, lang), found in groups.items():
        missing  = expected_variants - found
        extra    = found - expected_variants
        if missing:
            logger.warning(
                "prompt_id=%s lang=%s — missing variants: %s", pid, lang, missing
            )
            bad += 1
        if extra:
            logger.warning(
                "prompt_id=%s lang=%s — unexpected variants: %s", pid, lang, extra
            )
            bad += 1
    return bad


def check_empty_responses(records: list) -> int:
    """Return count of records with empty or whitespace-only responses."""
    bad = 0
    for rec in records:
        if not str(rec.get("response", "")).strip():
            logger.warning(
                "Empty response — prompt_id=%s lang=%s variant=%s",
                rec.get("prompt_id"), rec.get("language"), rec.get("variant_type"),
            )
            bad += 1
    return bad


def check_languages(records: list, valid_languages: set) -> int:
    """Return count of records whose language code is not in valid_languages."""
    bad = 0
    seen_invalid = set()
    for rec in records:
        lang = rec.get("language")
        if lang not in valid_languages and lang not in seen_invalid:
            logger.warning("Unexpected language code: '%s'", lang)
            seen_invalid.add(lang)
            bad += 1
    return bad


# ---------------------------------------------------------------------------
# Load helper
# ---------------------------------------------------------------------------

def load_records(path: str) -> list:
    p = str(path)
    with open(p, "r", encoding="utf-8") as fh:
        if p.endswith(".jsonl"):
            return [json.loads(line) for line in fh if line.strip()]
        return json.load(fh)


# ---------------------------------------------------------------------------
# Main validation runner
# ---------------------------------------------------------------------------

def validate(path: str, strict: bool = False) -> bool:
    """
    Run all checks on the dataset at ``path``.

    Parameters
    ----------
    path   : str   Path to raw_outputs.json or .jsonl
    strict : bool  If True, treat empty responses as a critical failure.

    Returns
    -------
    bool — True if dataset passes all checks, False otherwise.
    """
    if not Path(path).exists():
        logger.error("Dataset file not found: %s", path)
        return False

    records = load_records(path)
    logger.info("Loaded %d records from %s", len(records), path)

    if not records:
        logger.error("Dataset is empty.")
        return False

    issues = 0

    # 1. Required fields
    n = check_required_fields(records)
    if n:
        logger.error("[CRITICAL] %d record(s) have missing required fields.", n)
        issues += n

    # 2. Variant counts
    n = check_variant_counts(records, EXPECTED_VARIANTS)
    if n:
        logger.error("[CRITICAL] %d group(s) have incorrect variant sets.", n)
        issues += n

    # 3. Empty responses
    n = check_empty_responses(records)
    if n:
        level = logger.error if strict else logger.warning
        level("[%s] %d record(s) have empty responses.", "CRITICAL" if strict else "WARNING", n)
        if strict:
            issues += n

    # 4. Language codes
    n = check_languages(records, EXPECTED_LANGUAGES)
    if n:
        logger.warning("[WARNING] %d record(s) have unexpected language codes.", n)

    # Summary
    total_groups = len({(r.get("prompt_id"), r.get("language")) for r in records})
    logger.info("─" * 50)
    logger.info("Total records   : %d", len(records))
    logger.info("Unique groups   : %d  (prompt × language)", total_groups)
    logger.info("Unique prompts  : %d", len({r.get("prompt_id") for r in records}))

    if issues == 0:
        logger.info("Validation PASSED — no critical issues found.")
        return True
    else:
        logger.error("Validation FAILED — %d critical issue(s) found.", issues)
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a raw_outputs dataset before running analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", default="data/raw_outputs.json",
        help="Path to raw_outputs.json or .jsonl",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Treat empty responses as a critical failure (exit code 1).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    passed = validate(args.input, strict=args.strict)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

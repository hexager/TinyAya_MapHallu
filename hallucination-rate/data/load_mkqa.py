"""
Load MKQA dataset and sample N examples with queries in specified languages.
"""

import gzip
import json
import random
import urllib.request
from typing import List

# Try datasets first; fallback to direct download
try:
    from datasets import load_dataset
    _HAS_DATASETS = True
except ImportError:
    _HAS_DATASETS = False

MKQA_URL = "https://github.com/apple/ml-mkqa/raw/main/dataset/mkqa.jsonl.gz"

_EXCLUDED_ANSWER_TYPES = {"unanswerable"}


def _has_valid_gold_answer(ans: list) -> bool:
    """
    Return True if this MKQA answer list contains at least one usable gold string.

    We exclude items like:
      - {"type": "unanswerable", "text": null}
      - {"type": "long_answer", "text": null}
    i.e., any answer entry whose text is missing/empty, and any explicitly unanswerable items.
    """
    if not isinstance(ans, list) or not ans:
        return False
    for a in ans:
        if not isinstance(a, dict):
            continue
        if (a.get("type") or "").strip() in _EXCLUDED_ANSWER_TYPES:
            return False
    # At least one non-empty text or alias
    for a in ans:
        if not isinstance(a, dict):
            continue
        t = a.get("text")
        if isinstance(t, str) and t.strip():
            return True
        aliases = a.get("aliases") or []
        if any(isinstance(x, str) and x.strip() for x in aliases):
            return True
    return False


def _load_mkqa_raw(max_examples: int | None = None) -> List[dict]:
    """Load MKQA examples (full rows with queries + answers). Optionally cap at max_examples."""
    all_rows: List[dict] = []

    if _HAS_DATASETS:
        try:
            ds = load_dataset("apple/mkqa", split="train", trust_remote_code=True)
            for i, row in enumerate(ds):
                if max_examples is not None and i >= max_examples:
                    break
                all_rows.append(dict(row))
            if all_rows:
                return all_rows
        except Exception:
            pass

    # Fallback: direct download
    with urllib.request.urlopen(MKQA_URL) as response:
        with gzip.open(response, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                all_rows.append(obj)
                if max_examples is not None and len(all_rows) >= max_examples:
                    break
    return all_rows


def load_mkqa_samples(
    n: int = 500,
    seed: int = 42,
) -> List[dict]:
    """
    Load MKQA and return a random sample of n examples.

    Each item has: example_id, query (en), queries (dict lang -> str), answers (dict lang -> list).
    """
    raw = _load_mkqa_raw()
    random.seed(seed)
    sampled = random.sample(raw, min(n, len(raw)))
    return sampled


def get_queries_for_languages(
    samples: List[dict],
    languages: List[str],
) -> List[dict]:
    """
    From loaded MKQA samples, build one row per (example, language) with query text and ref answers.

    Returns list of:
      {
        "example_id": int,
        "language": str,
        "query": str,
        "answers": list (from MKQA answers[lang]),
      }
    """
    rows = []
    for s in samples:
        queries = s.get("queries") or {}
        answers = s.get("answers") or {}
        for lang in languages:
            q = (queries.get(lang) or queries.get("en") or "").strip()
            if not q:
                continue
            ans = answers.get(lang) or answers.get("en") or []
            if not _has_valid_gold_answer(ans):
                continue
            rows.append({
                "example_id": s.get("example_id"),
                "language": lang,
                "query": q,
                "answers": ans,
            })
    return rows

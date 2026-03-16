"""
src/collect_data.py — Stage 1: Data collection for the PSS pipeline.

Loads MKQA prompts, generates four task-preserving variants per prompt per
language, queries tiny-aya via the Cohere API (through src/helpers.py), and
writes all inputs/outputs incrementally to data/raw_outputs.jsonl.

All model calls go through:
    from src.helpers import query_model, get_text_from_response

Usage
-----
    python src/collect_data.py
    python src/collect_data.py --n_prompts 100 --model tiny-aya-global
    python src/collect_data.py --logprobs          # enable logprob storage
"""

import argparse
import json
import logging
import random
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

from datasets import load_dataset

# Ensure project root is on sys.path when run as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DATA_DIR,
    ERRORS_LOG_FILE,
    EXPERIMENT_VERSION,
    LANGUAGE_INSTRUCTIONS,
    LANGUAGES,
    LOGS_DIR,
    MODEL_ID,
    N_PROMPTS,
    PAUSE_FILE,
    PROMPT_VARIANTS,
    RANDOM_SEED,
    TEMPERATURE,
)
from prompt_variants import generate_variants
from src.helpers import get_logprobs_from_response, get_text_from_response, query_model

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RAW_OUTPUTS_FILE = "data/raw_outputs.jsonl"

_MAX_RETRIES = 2
_RETRY_DELAY = 2


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def _setup_dirs() -> None:
    for d in (DATA_DIR, "analysis", "plots", LOGS_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)


def _setup_file_logging(run_id: str) -> None:
    log_path = Path(LOGS_DIR) / f"collect_{run_id}.log"
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
    ))
    logging.getLogger().addHandler(fh)
    logger.info("Logging to file: %s", log_path)


# ---------------------------------------------------------------------------
# Pause / resume
# ---------------------------------------------------------------------------

def load_completed_keys(path: str) -> set:
    """
    Return (prompt_id, language, variant_type) tuples already written to disk.
    Enables crash-safe resume — completed records are skipped on re-run.
    """
    if not Path(path).exists():
        return set()
    completed, bad = set(), 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                completed.add((rec["prompt_id"], rec["language"], rec["variant_type"]))
            except (json.JSONDecodeError, KeyError):
                bad += 1
    if bad:
        logger.warning("Skipped %d malformed lines in existing JSONL.", bad)
    return completed


def check_pause() -> None:
    """Block until data/PAUSE is deleted. Create that file to pause the run."""
    if not Path(PAUSE_FILE).exists():
        return
    logger.info("PAUSED — delete '%s' to resume.", PAUSE_FILE)
    while Path(PAUSE_FILE).exists():
        time.sleep(10)
        logger.info("Still paused…")
    logger.info("Resumed.")


# ---------------------------------------------------------------------------
# MKQA loading
# ---------------------------------------------------------------------------

def load_mkqa_prompts(n: int, seed: int) -> List[dict]:
    """
    Sample ``n`` English queries from MKQA.

    Tries datasets library first, falls back to direct GitHub download.
    """
    logger.info("Loading MKQA …")
    queries: List[str] = []

    try:
        ds = load_dataset("apple/mkqa", split="train", trust_remote_code=True)
        queries = [r["query"] for r in ds if r.get("query", "").strip()]
        logger.info("Loaded %d queries via datasets library.", len(queries))
    except Exception as e:
        logger.warning("datasets library failed (%s). Trying direct download …", e)

    if not queries:
        import gzip, urllib.request
        url = "https://github.com/apple/ml-mkqa/raw/main/dataset/mkqa.jsonl.gz"
        logger.info("Fetching %s …", url)
        with urllib.request.urlopen(url) as resp:
            with gzip.open(resp, "rt", encoding="utf-8") as fh:
                for line in fh:
                    q = json.loads(line.strip()).get("query", "").strip()
                    if q:
                        queries.append(q)
        logger.info("Loaded %d queries via direct download.", len(queries))

    if not queries:
        raise RuntimeError("Could not load MKQA via any strategy.")

    random.seed(seed)
    sampled = random.sample(queries, min(n, len(queries)))
    return [{"prompt_id": i, "base_prompt": q} for i, q in enumerate(sampled)]


# ---------------------------------------------------------------------------
# Error logging
# ---------------------------------------------------------------------------

def _log_error(pid: int, lang: str, variant: str, msg: str) -> None:
    Path(ERRORS_LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt_id": pid, "language": lang, "variant_type": variant, "error": msg,
    }
    with open(ERRORS_LOG_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Model call with retry  (uses src.helpers.query_model)
# ---------------------------------------------------------------------------

def _call_with_retry(
    prompt: str,
    model: str,
    temp: float,
    use_logprobs: bool,
    retries: int = _MAX_RETRIES,
) -> Tuple[str, object]:
    """
    Call query_model() with retries.

    Returns
    -------
    (response_text, logprobs_or_None)
    """
    last_exc: Exception = RuntimeError("No attempts made.")
    for attempt in range(1 + retries):
        try:
            raw = query_model(query=prompt, model=model, temp=temp, logprobs=use_logprobs)
            text     = get_text_from_response(raw)
            logprobs = get_logprobs_from_response(raw) if use_logprobs else None
            return text, logprobs
        except Exception as exc:
            last_exc = exc
            logger.warning("Attempt %d/%d failed: %s", attempt + 1, 1 + retries, exc)
            if attempt < retries:
                time.sleep(_RETRY_DELAY)
    raise RuntimeError(f"All {1 + retries} attempt(s) failed.") from last_exc


# ---------------------------------------------------------------------------
# Core collection loop
# ---------------------------------------------------------------------------

def collect(
    prompts: List[dict],
    languages: List[str],
    model: str,
    temp: float,
    use_logprobs: bool,
    jsonl_fh,
    completed_keys: set,
) -> Tuple[int, int]:
    """
    Iterate every (prompt, language, variant) triplet, call the model via
    src.helpers.query_model, and write each record immediately (crash-safe).

    Record schema
    -------------
    prompt_id, prompt_text, language, variant_type, model, temperature,
    response, response_length[, logprobs]
    """
    total = len(prompts) * len(languages) * len(PROMPT_VARIANTS)
    written, skipped = 0, 0

    for lang in languages:
        suffix = LANGUAGE_INSTRUCTIONS.get(lang, "")

        for item in prompts:
            check_pause()

            pid      = item["prompt_id"]
            base_en  = item["base_prompt"]
            lang_base = (base_en + suffix).strip()

            all_variants = [
                {"variant_type": "base", "variant_prompt": lang_base}
            ] + generate_variants(lang_base)

            for v in all_variants:
                vtype  = v["variant_type"]
                vprompt = v["variant_prompt"]

                if (pid, lang, vtype) in completed_keys:
                    skipped += 1
                    continue

                logger.info("[%4d/%d]  lang=%-3s  pid=%3d  variant=%-12s",
                            written + skipped + 1, total, lang, pid, vtype)

                try:
                    text, logprobs = _call_with_retry(vprompt, model, temp, use_logprobs)
                except RuntimeError as exc:
                    logger.error("Giving up on pid=%d lang=%s variant=%s: %s",
                                 pid, lang, vtype, exc)
                    _log_error(pid, lang, vtype, str(exc))
                    text, logprobs = "", None

                record: dict = {
                    "prompt_id":       pid,
                    "prompt_text":     vprompt,
                    "language":        lang,
                    "variant_type":    vtype,
                    "model":           model,
                    "temperature":     temp,
                    "response":        text,
                    "response_length": len(text.split()),
                }
                if use_logprobs:
                    record["logprobs"] = logprobs

                jsonl_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                jsonl_fh.flush()
                written += 1

    return written, skipped


# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------

def _write_run_metadata(run_id: str, model: str, n_prompts: int) -> None:
    metadata = {
        "run_id":               run_id,
        "model_id":             model,
        "languages":            LANGUAGES,
        "num_prompts":          n_prompts,
        "num_variants":         len(PROMPT_VARIANTS),
        "total_expected_calls": n_prompts * len(LANGUAGES) * len(PROMPT_VARIANTS),
        "experiment_version":   EXPERIMENT_VERSION,
        "timestamp":            datetime.now(timezone.utc).isoformat(),
    }
    p = Path("analysis/run_metadata.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Run metadata saved  →  %s", p)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect PSS data via Cohere API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n_prompts", type=int, default=N_PROMPTS)
    parser.add_argument("--model",     default=MODEL_ID)
    parser.add_argument("--dataset",   default="mkqa")
    parser.add_argument("--logprobs",  action="store_true",
                        help="Store logprobs in each record (disabled by default).")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _setup_dirs()

    run_id = (
        f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
        f"_{uuid.uuid4().hex[:8]}"
    )
    _setup_file_logging(run_id)

    logger.info("=" * 60)
    logger.info("PSS Data Collection  —  src/collect_data.py")
    logger.info("Run ID   : %s", run_id)
    logger.info("Model    : %s", args.model)
    logger.info("Prompts  : %d", args.n_prompts)
    logger.info("Languages: %s", LANGUAGES)
    logger.info("Logprobs : %s", args.logprobs)
    logger.info("=" * 60)

    _write_run_metadata(run_id, args.model, args.n_prompts)

    prompts        = load_mkqa_prompts(args.n_prompts, RANDOM_SEED)
    completed_keys = load_completed_keys(RAW_OUTPUTS_FILE)

    if completed_keys:
        logger.info("Resuming — %d records already on disk.", len(completed_keys))

    written, skipped, paused = 0, 0, False
    with open(RAW_OUTPUTS_FILE, "a", encoding="utf-8") as fh:
        try:
            written, skipped = collect(
                prompts, LANGUAGES, args.model, TEMPERATURE,
                args.logprobs, fh, completed_keys,
            )
        except KeyboardInterrupt:
            paused = True

    if paused:
        logger.info("Run paused. Re-run to resume automatically.")
    else:
        logger.info("Collection complete.")

    logger.info("Written: %d  |  Skipped: %d", written, skipped)


if __name__ == "__main__":
    main()

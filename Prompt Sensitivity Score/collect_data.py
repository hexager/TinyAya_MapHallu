"""
collect_data.py — Stage 1: Data collection for the PSS pipeline.

Loads MKQA base prompts, generates four task-preserving variants per prompt
per language, queries tiny-aya via the Cohere API, and writes every
input/output to ``data/raw_outputs.json`` (full) and ``data/run_summary.csv``.

Pipeline separation
-------------------
This script does ONLY data collection.  No metric computation and no plotting
happen here.  Downstream scripts read ``raw_outputs.json``:
    • analysis/compute_pss.py  — compute PSS metrics
    • plots/plot_pss.py        — visualise results

Usage
-----
    python collect_data.py --dataset mkqa --n_prompts 100

    # Smaller test run:
    python collect_data.py --n_prompts 5
"""

import argparse
import csv
import json
import logging
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

from datasets import load_dataset

from config import (
    CSV_SUMMARY_FILE,
    DATA_DIR,
    ERRORS_LOG_FILE,
    EXPERIMENT_VERSION,
    LANGUAGE_INSTRUCTIONS,
    LANGUAGES,
    LOGS_DIR,
    MAX_TOKENS,
    MODEL_ID,
    N_PROMPTS,
    PAUSE_FILE,
    PROMPT_VARIANTS,
    RANDOM_SEED,
    RAW_OUTPUTS_FILE,
    TEMPERATURE,
)
from model_client import ModelClient
from prompt_variants import generate_variants

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Number of retry attempts on model call failure (1 initial + N retries).
_MAX_RETRIES = 2
_RETRY_DELAY = 2  # seconds between retries


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def _setup_dirs() -> None:
    """Create output directories if they do not already exist."""
    for d in (DATA_DIR, "analysis", "plots", LOGS_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)


def _setup_file_logging(run_id: str) -> None:
    """Add a per-run file handler so every log line is persisted to logs/."""
    log_path = Path(LOGS_DIR) / f"collect_{run_id}.log"
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
    ))
    logging.getLogger().addHandler(fh)
    logger.info("Logging to file: %s", log_path)


# ---------------------------------------------------------------------------
# Pause / resume helpers
# ---------------------------------------------------------------------------

def load_completed_keys(jsonl_path: str) -> set:
    """
    Read an existing JSONL file and return the set of already-collected keys.

    Each key is a (prompt_id, language, variant_type) tuple.  The collection
    loop uses this set to skip records that are already on disk, enabling
    seamless resume after a pause or crash.

    Parameters
    ----------
    jsonl_path : str  Path to raw_outputs.jsonl (may not exist yet).

    Returns
    -------
    set of (int, str, str) — empty set if the file does not exist.
    """
    if not Path(jsonl_path).exists():
        return set()

    completed = set()
    skipped_lines = 0
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                completed.add((rec["prompt_id"], rec["language"], rec["variant_type"]))
            except (json.JSONDecodeError, KeyError):
                skipped_lines += 1

    if skipped_lines:
        logger.warning("Skipped %d malformed lines in existing JSONL.", skipped_lines)
    return completed


def check_pause() -> None:
    """
    File-based pause mechanism.

    If the file ``data/PAUSE`` exists, the collection loop will block here,
    logging a message every 10 seconds, until the file is deleted.

    To pause:  create the file  →  touch data/PAUSE   (or right-click → New File)
    To resume: delete the file  →  del data/PAUSE
    """
    if not Path(PAUSE_FILE).exists():
        return                          # fast path — no file, no overhead

    logger.info("─" * 50)
    logger.info("PAUSED  —  delete '%s' to resume.", PAUSE_FILE)
    logger.info("─" * 50)
    while Path(PAUSE_FILE).exists():
        time.sleep(10)
        logger.info("Still paused  —  waiting for '%s' to be deleted …", PAUSE_FILE)
    logger.info("Resumed.")


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_mkqa_prompts(n: int, seed: int) -> List[dict]:
    """
    Load the MKQA dataset and return a random sample of ``n`` English queries.

    Tries two strategies in order:
    1. ``datasets.load_dataset`` with ``trust_remote_code=True``
       (works with datasets < 3.0 which still supports loading scripts).
    2. Direct file download via ``huggingface_hub.hf_hub_download``
       (works with datasets >= 3.0 where loading scripts are removed).

    Parameters
    ----------
    n    : int  Number of base prompts to sample.
    seed : int  Random seed for reproducible sampling.

    Returns
    -------
    list of {"prompt_id": int, "base_prompt": str}
    """
    logger.info("Downloading / loading MKQA from HuggingFace …")

    all_queries: List[str] = []

    # ------------------------------------------------------------------
    # Strategy 1: datasets library with trust_remote_code
    # (works on datasets < 3.0.0; fails silently on newer versions)
    # ------------------------------------------------------------------
    try:
        ds = load_dataset("apple/mkqa", split="train", trust_remote_code=True)
        all_queries = [
            row["query"] for row in ds if row.get("query") and row["query"].strip()
        ]
        logger.info("MKQA loaded via datasets library: %d queries.", len(all_queries))
    except Exception as e:
        logger.warning(
            "datasets library load failed (%s). "
            "Trying direct file download (requires datasets >= 3.0 or huggingface_hub) …",
            e,
        )

    # ------------------------------------------------------------------
    # Strategy 2: download raw JSONL file directly via huggingface_hub
    # MKQA stores questions in mkqa.jsonl.gz on the HF repo.
    # ------------------------------------------------------------------
    if not all_queries:
        import gzip
        import urllib.request

        # Data lives in Apple's GitHub repo (confirmed from mkqa.py loading script).
        url = "https://github.com/apple/ml-mkqa/raw/main/dataset/mkqa.jsonl.gz"
        logger.info("Fetching mkqa.jsonl.gz from %s …", url)

        with urllib.request.urlopen(url) as response:
            with gzip.open(response, "rt", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    q = obj.get("query", "").strip()
                    if q:
                        all_queries.append(q)
        logger.info("MKQA loaded via direct download: %d queries.", len(all_queries))

    if not all_queries:
        raise RuntimeError(
            "Could not load MKQA dataset via any strategy.\n"
            "If using datasets >= 3.0, ensure huggingface_hub is installed:\n"
            "  pip install huggingface_hub\n"
            "Or downgrade datasets:\n"
            "  pip install 'datasets<3.0.0'"
        )

    random.seed(seed)
    sampled = random.sample(all_queries, min(n, len(all_queries)))

    return [{"prompt_id": i, "base_prompt": q} for i, q in enumerate(sampled)]


# ---------------------------------------------------------------------------
# Error logging
# ---------------------------------------------------------------------------

def _log_error(
    prompt_id: int,
    language: str,
    variant_type: str,
    error_msg: str,
) -> None:
    """Append a JSON line to the errors log without interrupting the main loop."""
    Path(ERRORS_LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "prompt_id":   prompt_id,
        "language":    language,
        "variant_type": variant_type,
        "error":       error_msg,
    }
    with open(ERRORS_LOG_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Model call with retry
# ---------------------------------------------------------------------------

def _call_with_retry(
    client: ModelClient,
    prompt: str,
    retries: int = _MAX_RETRIES,
) -> Tuple[str, int]:
    """
    Call ``client.generate(prompt)`` with up to ``retries`` retry attempts.

    Returns
    -------
    (response_text, token_count)

    Raises
    ------
    RuntimeError  if all attempts fail.
    """
    last_exc: Exception = RuntimeError("No attempts made.")
    for attempt in range(1 + retries):
        try:
            return client.generate(prompt)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning(
                "Attempt %d/%d failed: %s", attempt + 1, 1 + retries, exc
            )
            if attempt < retries:
                time.sleep(_RETRY_DELAY)
    raise RuntimeError(f"All {1 + retries} attempt(s) failed.") from last_exc


# ---------------------------------------------------------------------------
# Core collection loop
# ---------------------------------------------------------------------------

def collect(
    prompts: List[dict],
    client: ModelClient,
    languages: List[str],
    run_id: str,
    jsonl_fh,                    # open file handle for incremental JSONL writes
    csv_writer,                  # csv.DictWriter already past writeheader()
    completed_keys: set,         # (prompt_id, language, variant_type) already on disk
) -> Tuple[int, int]:
    """
    Iterate over every (prompt, language, variant) triplet, call the model,
    and write each record to disk immediately after the API call returns.

    Records whose key (prompt_id, language, variant_type) is already present
    in ``completed_keys`` are silently skipped — this is the resume mechanism.

    Pause: pressing Ctrl+C raises KeyboardInterrupt which propagates to
    ``main()``.  Because writes are incremental, all records collected so far
    are already safely on disk.  Re-running the script resumes automatically.

    Parameters
    ----------
    prompts          : output of load_mkqa_prompts
    client           : initialised ModelClient
    languages        : list of language codes
    run_id           : unique string identifying this collection run
    jsonl_fh         : writable file handle for raw_outputs.jsonl
    csv_writer       : csv.DictWriter already past writeheader()
    completed_keys   : set of (prompt_id, language, variant_type) to skip

    Returns
    -------
    (new_written, skipped) : counts of newly written and skipped records
    """
    total_calls = len(prompts) * len(languages) * 5   # base + 4 variants
    new_written = 0
    skipped     = 0

    for lang in languages:
        lang_suffix = LANGUAGE_INSTRUCTIONS.get(lang, "")

        for item in prompts:
            # Check for file-based pause before each prompt group.
            # Create data/PAUSE to pause; delete it to resume.
            check_pause()

            pid     = item["prompt_id"]
            base_en = item["base_prompt"]

            lang_base = (base_en + lang_suffix).strip()

            all_variants = [
                {"variant_type": "base", "variant_prompt": lang_base}
            ] + generate_variants(lang_base)

            for v in all_variants:
                variant_type   = v["variant_type"]
                variant_prompt = v["variant_prompt"]

                # --- Resume: skip if already collected in a previous run ---
                if (pid, lang, variant_type) in completed_keys:
                    skipped += 1
                    continue

                progress = new_written + skipped + 1
                logger.info(
                    "[%4d/%d]  lang=%-3s  pid=%3d  variant=%-12s",
                    progress, total_calls, lang, pid, variant_type,
                )

                # Model call (with retry on failure)
                try:
                    response, n_tokens = _call_with_retry(client, variant_prompt)
                except RuntimeError as exc:
                    logger.error(
                        "Giving up on pid=%d lang=%s variant=%s: %s",
                        pid, lang, variant_type, exc,
                    )
                    _log_error(pid, lang, variant_type, str(exc))
                    response = ""
                    n_tokens = 0

                record = {
                    "run_id":          run_id,
                    "prompt_id":       pid,
                    "language":        lang,
                    "variant_type":    variant_type,
                    "base_prompt":     base_en,
                    "variant_prompt":  variant_prompt,
                    "model":           MODEL_ID,
                    "temperature":     TEMPERATURE,
                    "max_tokens":      MAX_TOKENS,
                    "response":        response,
                    "response_tokens": n_tokens,
                    "timestamp":       datetime.now(timezone.utc).isoformat(),
                }

                # Incremental write + flush — crash-safe.
                jsonl_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                jsonl_fh.flush()
                csv_writer.writerow({k: record[k] for k in _CSV_FIELDS})
                new_written += 1

    return new_written, skipped


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "run_id", "prompt_id", "language", "variant_type",
    "base_prompt", "response_tokens", "timestamp",
]


def open_output_files(jsonl_path: str, csv_path: str):
    """
    Open both output files for incremental writing and return their handles.

    The JSONL file is opened in *append* mode so re-running after a crash
    adds new records without overwriting prior ones.

    Returns
    -------
    (jsonl_fh, csv_fh, csv_writer)
        Callers are responsible for closing jsonl_fh and csv_fh when done.
    """
    jsonl_fh = open(jsonl_path, "a", encoding="utf-8")   # append — crash-safe

    # Write CSV header only if the file is new (size == 0).
    csv_fh = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_fh, fieldnames=_CSV_FIELDS, extrasaction="ignore")
    if csv_fh.tell() == 0:
        writer.writeheader()

    logger.info("Incremental output: %s  |  %s", jsonl_path, csv_path)
    return jsonl_fh, csv_fh, writer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect PSS data via Cohere API (tiny-aya).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default="mkqa",
        help="HuggingFace dataset identifier.",
    )
    parser.add_argument(
        "--n_prompts",
        type=int,
        default=N_PROMPTS,
        help="Number of base prompts sampled from the dataset.",
    )
    parser.add_argument(
        "--model",
        default=MODEL_ID,
        help="Cohere model ID to use for generation.",
    )
    return parser.parse_args()


def _write_run_metadata(run_id: str, model_id: str, n_prompts: int) -> None:
    """Write experiment metadata to analysis/run_metadata.json."""
    metadata = {
        "run_id":               run_id,
        "model_id":             model_id,
        "languages":            LANGUAGES,
        "num_prompts":          n_prompts,
        "num_variants":         len(PROMPT_VARIANTS),
        "total_expected_calls": n_prompts * len(LANGUAGES) * len(PROMPT_VARIANTS),
        "experiment_version":   EXPERIMENT_VERSION,
        "timestamp":            datetime.now(timezone.utc).isoformat(),
    }
    meta_path = Path("analysis/run_metadata.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Run metadata saved  →  %s", meta_path)


def main() -> None:
    args = _parse_args()
    _setup_dirs()

    # Unique run identifier — used to correlate records from the same batch.
    run_id = (
        f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
        f"_{uuid.uuid4().hex[:8]}"
    )
    _setup_file_logging(run_id)

    # Allow --model to override config MODEL_ID at runtime.
    active_model = args.model

    logger.info("=" * 60)
    logger.info("PSS Data Collection  (Cohere API)")
    logger.info("Run ID   : %s", run_id)
    logger.info("Model    : %s", active_model)
    logger.info("Dataset  : %s", args.dataset)
    logger.info("Prompts  : %d", args.n_prompts)
    logger.info("Languages: %s", LANGUAGES)
    logger.info("=" * 60)

    _write_run_metadata(run_id, active_model, args.n_prompts)

    prompts = load_mkqa_prompts(args.n_prompts, RANDOM_SEED)
    logger.info("Sampled %d base prompts.", len(prompts))

    # --- Resume: load keys already on disk before opening files for writing ---
    completed_keys = load_completed_keys(RAW_OUTPUTS_FILE)
    if completed_keys:
        logger.info(
            "Resuming: %d records already collected — skipping those.",
            len(completed_keys),
        )
    else:
        logger.info("No existing records found — starting fresh.")

    client = ModelClient()

    jsonl_fh, csv_fh, csv_writer = open_output_files(RAW_OUTPUTS_FILE, CSV_SUMMARY_FILE)
    new_written, skipped, paused = 0, 0, False
    try:
        new_written, skipped = collect(
            prompts, client, LANGUAGES, run_id, jsonl_fh, csv_writer, completed_keys
        )
    except KeyboardInterrupt:
        # --- Pause: Ctrl+C — files are already flushed, nothing is lost ---
        paused = True
    finally:
        # Always close — records written so far are safe regardless of exit path.
        jsonl_fh.close()
        csv_fh.close()

    if paused:
        logger.info("")
        logger.info("Run paused.  Re-run the same command to resume automatically.")
        logger.info("Progress saved in: %s", RAW_OUTPUTS_FILE)
    else:
        logger.info("Collection complete.")

    logger.info(
        "New records written: %d  |  Skipped (already done): %d",
        new_written, skipped,
    )


if __name__ == "__main__":
    main()

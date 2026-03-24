"""
Run a model on a list of (query, ...) items, log every request/response to file, return results for plotting.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from tqdm import tqdm

from config import LOGS_DIR, RESULTS_DIR
from metrics import add_correctness_to_results, compute_hallucination_rate
from model_client import query_model

logger = logging.getLogger(__name__)


def run_evaluation(
    items: List[dict],
    model: str,
    experiment_name: str,
    run_id: str | None = None,
    log_dir: Path | None = None,
    results_dir: Path | None = None,
    retries: int = 2,
    delay_seconds: float = 1.0,
) -> List[dict]:
    """
    For each item, call the model with item["query"], log to file, collect (query, response, tokens, etc.).

    items: list of dicts with at least "query" key; can have "example_id", "language", "answers".
    model: Cohere model id (e.g. tiny-aya-global, tiny-aya-water).
    experiment_name: used for log/result subdirs.
    run_id: optional; default is timestamp.
    log_dir: optional; default LOGS_DIR / experiment_name.
    results_dir: optional; default RESULTS_DIR / experiment_name.

    Returns list of dicts: original item + response_text, output_tokens, latency_sec, model.
    """
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_dir = log_dir or (Path(LOGS_DIR) / experiment_name)
    results_dir = results_dir or (Path(RESULTS_DIR) / experiment_name)
    log_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"eval_{model}_{run_id}.jsonl"
    results: List[dict] = []

    for i, item in enumerate(tqdm(items, desc=f"{model}", unit="q")):
        query = item.get("query", "")
        if not query:
            results.append({**item, "response_text": "", "output_tokens": 0, "latency_sec": 0, "model": model, "error": "empty query"})
            continue

        last_exc = None
        for attempt in range(1 + retries):
            try:
                start = time.perf_counter()
                response_text, output_tokens = query_model(query, model=model)
                latency_sec = time.perf_counter() - start
                break
            except Exception as e:
                last_exc = e
                if attempt < retries:
                    time.sleep(delay_seconds)
        else:
            response_text = ""
            output_tokens = 0
            latency_sec = 0
            logger.warning("All retries failed for item %s: %s", i, last_exc)

        record = {
            **{k: v for k, v in item.items() if k in ("example_id", "language", "query", "answers")},
            "response_text": response_text,
            "output_tokens": output_tokens,
            "latency_sec": round(latency_sec, 4),
            "model": model,
            "index": i,
        }
        results.append(record)

        # Log each response to file (append JSONL)
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "experiment": experiment_name,
            "index": i,
            "example_id": item.get("example_id"),
            "language": item.get("language"),
            "query": query[:500],
            "response_text": response_text[:2000],
            "output_tokens": output_tokens,
            "latency_sec": record["latency_sec"],
        }
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    # Add correctness vs MKQA gold and compute hallucination rate
    add_correctness_to_results(results)
    summary = compute_hallucination_rate(results)
    summary["model"] = model
    summary["run_id"] = run_id
    summary["experiment_name"] = experiment_name

    # Write full results for this model/run
    results_file = results_dir / f"results_{model}_{run_id}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Write hallucination-rate summary
    summary_file = results_dir / f"summary_{model}_{run_id}.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(
        "Hallucination rate: %.2f%% (accuracy %.2f%%) — %s",
        summary["hallucination_rate"] * 100,
        summary["accuracy"] * 100,
        summary_file,
    )
    logger.info("Wrote %d results to %s and log to %s", len(results), results_file, log_file)
    return results

"""
Experiment 1: TinyAya Global vs TinyAya Water on a language OUTSIDE Asia-Pacific/Europe.

Goal: Test whether finetuning (Water) degrades performance on languages it was not
primarily trained on. Uses one out-of-region language (e.g. Arabic).
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from config import (
    DEFAULT_OUT_OF_REGION_LANG,
    LOGS_DIR,
    MKQA_NUM_SAMPLES,
    MKQA_SEED,
    MODELS,
    PLOTS_DIR,
    RESULTS_DIR,
)
from data import get_queries_for_languages, load_mkqa_samples
from evaluate import run_evaluation
from plotting import plot_hallucination_rate

EXPERIMENT_NAME = "exp1_out_of_region"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_file_logging(run_id: str) -> Path:
    log_dir = LOGS_DIR / EXPERIMENT_NAME
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"run_{run_id}.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(fh)
    logger.info("Logging to %s", log_file)
    return log_file


def run(
    language: str | None = None,
    num_samples: int | None = None,
    run_id: str | None = None,
) -> dict:
    """
    Run Experiment 1: load MKQA, filter to one out-of-region language, evaluate both models, plot.
    """
    language = language or DEFAULT_OUT_OF_REGION_LANG
    num_samples = num_samples or MKQA_NUM_SAMPLES
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    setup_file_logging(run_id)
    logger.info("Experiment 1: out-of-region language=%s, samples=%d, run_id=%s", language, num_samples, run_id)

    samples = load_mkqa_samples(n=num_samples, seed=MKQA_SEED)
    items = get_queries_for_languages(samples, [language])
    logger.info("Loaded %d query items for language %s", len(items), language)

    results_by_model = {}
    for model in MODELS:
        results = run_evaluation(
            items,
            model=model,
            experiment_name=EXPERIMENT_NAME,
            run_id=run_id,
        )
        results_by_model[model] = results

    # Plot hallucination rate only
    plots_dir = PLOTS_DIR / EXPERIMENT_NAME
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_hallucination_rate(results_by_model, EXPERIMENT_NAME, plots_dir=plots_dir, run_id=run_id)

    logger.info("Experiment 1 done. Results and hallucination-rate summary in %s, plot in %s", RESULTS_DIR / EXPERIMENT_NAME, plots_dir)
    return {"run_id": run_id, "language": language, "n_items": len(items), "results_by_model": results_by_model}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Experiment 1: Global vs Water on out-of-region language")
    parser.add_argument("--language", default=DEFAULT_OUT_OF_REGION_LANG, help="Out-of-region language code (e.g. ar, he)")
    parser.add_argument("--num-samples", type=int, default=MKQA_NUM_SAMPLES, help="MKQA samples")
    parser.add_argument("--run-id", type=str, default=None, help="Run ID (default: timestamp)")
    args = parser.parse_args()
    run(language=args.language, num_samples=args.num_samples, run_id=args.run_id)
    sys.exit(0)

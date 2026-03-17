"""
Experiment 2: TinyAya Global vs TinyAya Water on Asia-Pacific / Europe languages.

Goal: Compare Global vs Water on the region where Water is strongest (Asia-Pacific and Europe).
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from config import (
    IN_REGION_LANGUAGES,
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

EXPERIMENT_NAME = "exp2_in_region"

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
    languages: list[str] | None = None,
    num_samples: int | None = None,
    run_id: str | None = None,
) -> dict:
    """
    Run Experiment 2: load MKQA, filter to in-region languages, evaluate both models, plot.
    """
    languages = languages or IN_REGION_LANGUAGES
    num_samples = num_samples or MKQA_NUM_SAMPLES
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    setup_file_logging(run_id)
    logger.info("Experiment 2: in-region languages=%s, samples=%d, run_id=%s", len(languages), num_samples, run_id)

    samples = load_mkqa_samples(n=num_samples, seed=MKQA_SEED)
    items = get_queries_for_languages(samples, languages)
    # Cap at num_samples evaluation items for comparable cost to exp1
    if len(items) > num_samples:
        import random
        random.seed(MKQA_SEED)
        items = random.sample(items, num_samples)
    logger.info("Loaded %d query items across %d languages", len(items), len(languages))

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

    logger.info("Experiment 2 done. Results and hallucination-rate summary in %s, plot in %s", RESULTS_DIR / EXPERIMENT_NAME, plots_dir)
    return {"run_id": run_id, "languages": languages, "n_items": len(items), "results_by_model": results_by_model}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Experiment 2: Global vs Water on Asia-Pacific/Europe")
    parser.add_argument("--num-samples", type=int, default=MKQA_NUM_SAMPLES, help="MKQA samples")
    parser.add_argument("--run-id", type=str, default=None, help="Run ID (default: timestamp)")
    parser.add_argument("--languages", type=str, nargs="+", default=None, help="Override in-region languages")
    args = parser.parse_args()
    run(num_samples=args.num_samples, run_id=args.run_id, languages=args.languages)
    sys.exit(0)

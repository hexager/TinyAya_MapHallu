"""
Entrypoint to run TinyAya Global vs Water experiments.

Usage:
  python run_experiments.py --experiment 1
  python run_experiments.py --experiment 2
  python run_experiments.py --experiment 1 2
  python run_experiments.py --experiment 1 --language he --num-samples 200
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import LOGS_DIR, OUTPUT_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run TinyAya Global vs Water experiments (MKQA, logging, plots)"
    )
    parser.add_argument(
        "--experiment",
        type=int,
        nargs="+",
        choices=[1, 2],
        default=[1, 2],
        help="Which experiment(s) to run: 1=out-of-region, 2=in-region",
    )
    parser.add_argument("--language", type=str, default=None, help="Exp1 only: out-of-region language code (e.g. ar, he)")
    parser.add_argument("--num-samples", type=int, default=None, help="MKQA sample size (default: 500)")
    parser.add_argument("--run-id", type=str, default=None, help="Optional run ID for this run")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (LOGS_DIR / "run_experiments").mkdir(parents=True, exist_ok=True)

    if 1 in args.experiment:
        from experiment_1_out_of_region import run as run_exp1
        run_exp1(
            language=args.language,
            num_samples=args.num_samples,
            run_id=args.run_id,
        )
    if 2 in args.experiment:
        from experiment_2_in_region import run as run_exp2
        run_exp2(
            num_samples=args.num_samples,
            run_id=args.run_id,
        )

    logger.info("Done. Check output/ for logs, results, and plots.")


if __name__ == "__main__":
    main()

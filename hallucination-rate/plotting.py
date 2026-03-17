"""
Plot hallucination rate: Global vs Water (bar chart).
"""

import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt

from config import PLOTS_DIR
from metrics import compute_hallucination_rate


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_hallucination_rate(
    results_by_model: dict[str, List[dict]],
    experiment_name: str,
    plots_dir: Path | None = None,
    run_id: str | None = None,
) -> Path:
    """
    Plot hallucination rate (and accuracy) as a bar chart: one bar per model.
    results_by_model: {"tiny-aya-global": [...], "tiny-aya-water": [...]}
    """
    plots_dir = plots_dir or (Path(PLOTS_DIR) / experiment_name)
    _ensure_dir(plots_dir)
    suffix = f"_{run_id}" if run_id else ""

    models = list(results_by_model.keys())
    rates = [compute_hallucination_rate(results_by_model[m])["hallucination_rate"] for m in models]
    accuracies = [compute_hallucination_rate(results_by_model[m])["accuracy"] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(f"TinyAya — {experiment_name}", fontsize=12)

    ax = axes[0]
    bars = ax.bar(models, [r * 100 for r in rates], color=["#2e86ab", "#a23b72"], alpha=0.85)
    ax.set_ylabel("Hallucination rate (%)")
    ax.set_title("Hallucination rate (1 − accuracy vs MKQA gold)")
    ax.set_ylim(0, 100)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.1f}%", ha="center", fontsize=10)

    ax = axes[1]
    bars = ax.bar(models, [a * 100 for a in accuracies], color=["#2e86ab", "#a23b72"], alpha=0.85)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy (match vs MKQA gold)")
    ax.set_ylim(0, 100)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.1f}%", ha="center", fontsize=10)

    plt.tight_layout()
    out_path = plots_dir / f"hallucination_rate{suffix}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def load_results_from_dirs(
    results_dir: Path,
    run_id: str | None = None,
) -> dict[str, List[dict]]:
    """Load results JSON files from results_dir. If run_id given, filter by that run."""
    results_by_model: dict[str, List[dict]] = {}
    for p in results_dir.iterdir():
        if p.suffix != ".json" or not p.name.startswith("results_"):
            continue
        rest = p.stem.replace("results_", "", 1)
        parts = rest.split("_")
        if len(parts) >= 3:
            file_run_id = "_".join(parts[-2:])
            model = "_".join(parts[:-2])
        else:
            file_run_id = ""
            model = rest
        if run_id and file_run_id != run_id:
            continue
        with open(p, encoding="utf-8") as f:
            results_by_model[model] = json.load(f)
    return results_by_model

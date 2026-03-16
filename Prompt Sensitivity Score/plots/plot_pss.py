"""
plots/plot_pss.py — Stage 4: Visualise Prompt Sensitivity Score results.

Reads only pre-computed CSVs.  Does NOT call the model or recompute metrics.

Inputs
------
analysis/pss_scores.csv       — per-row PSS scores (output of compute_pss_score.py)
analysis/unstable_prompts.csv — ranked instability table

Outputs
-------
plots/pss_distribution.png    — histogram of prompt_sensitivity_score
plots/pss_by_language.png     — boxplot of PSS by language
plots/semantic_vs_entity.png  — scatter: semantic_similarity vs entity_change_rate
plots/top_unstable_prompts.png — bar chart of top-20 most unstable prompts

Usage
-----
    python plots/plot_pss.py
    python plots/plot_pss.py --scores analysis/pss_scores.csv
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.05)

LANG_PALETTE = {"en": "#4C72B0", "es": "#DD8452", "hi": "#55A868", "ar": "#C44E52"}

PLOTS_DIR = Path("plots")


def _save(fig: plt.Figure, path: str) -> None:
    """Save figure and close it."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info("Saved  →  %s", path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 1 — PSS Distribution (histogram)
# ---------------------------------------------------------------------------

def plot_pss_distribution(df: pd.DataFrame, out: str) -> None:
    """
    Histogram of ``prompt_sensitivity_score`` across all scored rows.

    Annotates mean and median with vertical lines so researchers can
    quickly read overall stability at a glance.

    Parameters
    ----------
    df  : pss_scores DataFrame
    out : output file path
    """
    scores = df["prompt_sensitivity_score"].dropna()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(scores, bins=30, color="#4C72B0", edgecolor="white", alpha=0.85)

    mean_val   = scores.mean()
    median_val = scores.median()
    ax.axvline(mean_val,   color="#DD8452", linewidth=1.8, linestyle="--",
               label=f"Mean   {mean_val:.4f}")
    ax.axvline(median_val, color="#55A868", linewidth=1.8, linestyle=":",
               label=f"Median {median_val:.4f}")

    ax.set_title("Prompt Sensitivity Score — Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Prompt Sensitivity Score")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)

    _save(fig, out)


# ---------------------------------------------------------------------------
# Plot 2 — PSS by Language (boxplot)
# ---------------------------------------------------------------------------

def plot_pss_by_language(df: pd.DataFrame, out: str) -> None:
    """
    Boxplot of ``prompt_sensitivity_score`` grouped by language.

    Languages with all-null scores (ar, hi) are shown as empty to make
    their absence visible rather than hiding them entirely.

    Parameters
    ----------
    df  : pss_scores DataFrame
    out : output file path
    """
    languages = sorted(df["language"].unique())
    palette   = [LANG_PALETTE.get(l, "#888888") for l in languages]

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(
        data=df,
        x="language",
        y="prompt_sensitivity_score",
        order=languages,
        palette=palette,
        ax=ax,
        linewidth=1.2,
        fliersize=3,
    )
    ax.set_title("Prompt Sensitivity Score by Language", fontsize=13, fontweight="bold")
    ax.set_xlabel("Language")
    ax.set_ylabel("Prompt Sensitivity Score")

    # Annotate languages that have no scores.
    for i, lang in enumerate(languages):
        subset = df.loc[df["language"] == lang, "prompt_sensitivity_score"].dropna()
        if subset.empty:
            ax.text(i, 0.01, "no data\n(null ECR)", ha="center", va="bottom",
                    fontsize=8, color="grey", style="italic")

    _save(fig, out)


# ---------------------------------------------------------------------------
# Plot 3 — Semantic Similarity vs Entity Change Rate (scatter)
# ---------------------------------------------------------------------------

def plot_semantic_vs_entity(df: pd.DataFrame, out: str) -> None:
    """
    Scatter plot of semantic_similarity vs entity_change_rate, coloured
    by language.  Reveals whether semantic drift correlates with factual
    instability.

    Only rows where both values are non-null are plotted.

    Parameters
    ----------
    df  : pss_scores DataFrame
    out : output file path
    """
    plot_df = df.dropna(subset=["semantic_similarity", "entity_change_rate"])

    fig, ax = plt.subplots(figsize=(9, 6))

    for lang, grp in plot_df.groupby("language"):
        ax.scatter(
            grp["semantic_similarity"],
            grp["entity_change_rate"],
            label=lang,
            color=LANG_PALETTE.get(lang, "#888888"),
            alpha=0.6,
            s=30,
            edgecolors="none",
        )

    # Trend line across all plotted points.
    if len(plot_df) > 1:
        x = plot_df["semantic_similarity"].values
        y = plot_df["entity_change_rate"].values
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 200)
        ax.plot(x_line, m * x_line + b, color="black", linewidth=1.2,
                linestyle="--", label=f"Trend (slope {m:.3f})")

    ax.set_title("Semantic Similarity vs Entity Change Rate", fontsize=13, fontweight="bold")
    ax.set_xlabel("Semantic Similarity  (↑ = stable meaning)")
    ax.set_ylabel("Entity Change Rate  (↑ = factual drift)")
    ax.legend(fontsize=9, title="Language")

    _save(fig, out)


# ---------------------------------------------------------------------------
# Plot 4 — Top Unstable Prompts (bar chart)
# ---------------------------------------------------------------------------

def plot_top_unstable(unstable: pd.DataFrame, out: str, top_n: int = 20) -> None:
    """
    Horizontal bar chart of the top-N most unstable prompts ranked by
    ``avg_prompt_sensitivity_score``.

    Parameters
    ----------
    unstable : unstable_prompts DataFrame
    out      : output file path
    top_n    : how many prompts to display
    """
    data = unstable.head(top_n).copy()
    data = data.sort_values("avg_prompt_sensitivity_score", ascending=True)

    labels = [f"prompt {pid}" for pid in data["prompt_id"]]
    values = data["avg_prompt_sensitivity_score"].values

    # Colour bars by score intensity.
    norm   = plt.Normalize(values.min(), values.max())
    colors = plt.cm.YlOrRd(norm(values))  # type: ignore[attr-defined]

    fig, ax = plt.subplots(figsize=(10, max(5, len(data) * 0.38)))
    bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.7)

    # Value labels on each bar.
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + values.max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center", fontsize=8,
        )

    ax.set_title(f"Top {len(data)} Most Unstable Prompts",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Avg Prompt Sensitivity Score")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.margins(x=0.15)

    _save(fig, out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise PSS scores (reads CSVs only, no model calls).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scores",   default="analysis/pss_scores.csv",
                        help="Path to pss_scores.csv")
    parser.add_argument("--unstable", default="analysis/unstable_prompts.csv",
                        help="Path to unstable_prompts.csv")
    parser.add_argument("--out_dist", default="plots/pss_distribution.png")
    parser.add_argument("--out_lang", default="plots/pss_by_language.png")
    parser.add_argument("--out_scat", default="plots/semantic_vs_entity.png")
    parser.add_argument("--out_top",  default="plots/top_unstable_prompts.png")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    scores_df   = pd.read_csv(args.scores)
    unstable_df = pd.read_csv(args.unstable)
    logger.info("Loaded %d score rows, %d unstable-prompt rows.",
                len(scores_df), len(unstable_df))

    plot_pss_distribution(scores_df,   args.out_dist)
    plot_pss_by_language(scores_df,    args.out_lang)
    plot_semantic_vs_entity(scores_df, args.out_scat)
    plot_top_unstable(unstable_df,     args.out_top)

    logger.info("All plots saved to %s/", PLOTS_DIR)


if __name__ == "__main__":
    main()

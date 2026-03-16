"""
plots/plot_pss_clean.py — Clean PSS visualisation script.

Deletes any existing plot PNGs in plots/, then regenerates four figures
from pre-computed CSVs.  Does NOT call the model or recompute metrics.

Inputs
------
analysis/pss_scores.csv       — per-row PSS scores (output of compute_pss_score.py)
analysis/unstable_prompts.csv — ranked instability table

Outputs
-------
plots/pss_distribution.png     — histogram of prompt_sensitivity_score
plots/pss_by_language.png      — boxplot of PSS by language
plots/semantic_vs_entity.png   — scatter: semantic_similarity vs entity_change_rate
plots/top_unstable_prompts.png — bar chart, top-20 most unstable (most → least)

Usage
-----
    python plots/plot_pss_clean.py
    python plots/plot_pss_clean.py --scores analysis/pss_scores.csv
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
# Style
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.1)
plt.rcParams.update({
    "axes.titlesize":  15,
    "axes.labelsize":  13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
})

LANG_PALETTE = {"en": "#4C72B0", "es": "#DD8452", "hi": "#55A868", "ar": "#C44E52"}
PLOTS_DIR    = Path("plots")

PLOT_FILES = [
    "pss_distribution.png",
    "pss_by_language.png",
    "semantic_vs_entity.png",
    "top_unstable_prompts.png",
]


# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

def delete_old_plots() -> None:
    """Remove any existing PNG files in PLOTS_DIR before regenerating."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    removed = 0
    for fname in PLOT_FILES:
        p = PLOTS_DIR / fname
        if p.exists():
            p.unlink()
            logger.info("Deleted old plot  →  %s", p)
            removed += 1
    if removed == 0:
        logger.info("No old plots to delete.")


def _save(fig: plt.Figure, path: str) -> None:
    """Save figure and close it."""
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info("Saved  →  %s", path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 1 — PSS Distribution
# ---------------------------------------------------------------------------

def plot_pss_distribution(df: pd.DataFrame, out: str) -> None:
    scores = df["prompt_sensitivity_score"].dropna()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(scores, bins=30, color="#4C72B0", edgecolor="white", alpha=0.85)

    mean_val   = scores.mean()
    median_val = scores.median()
    ax.axvline(mean_val,   color="#DD8452", linewidth=2.0, linestyle="--",
               label=f"Mean   {mean_val:.4f}")
    ax.axvline(median_val, color="#55A868", linewidth=2.0, linestyle=":",
               label=f"Median {median_val:.4f}")

    ax.set_title("Prompt Sensitivity Score — Distribution", fontweight="bold")
    ax.set_xlabel("Prompt Sensitivity Score")
    ax.set_ylabel("Count")
    ax.legend()

    _save(fig, out)


# ---------------------------------------------------------------------------
# Plot 2 — PSS by Language
# ---------------------------------------------------------------------------

def plot_pss_by_language(df: pd.DataFrame, out: str) -> None:
    languages = sorted(df["language"].unique())
    palette   = [LANG_PALETTE.get(l, "#888888") for l in languages]

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(
        data=df,
        x="language",
        y="prompt_sensitivity_score",
        hue="language",
        order=languages,
        palette=dict(zip(languages, palette)),
        ax=ax,
        linewidth=1.4,
        fliersize=3,
        legend=False,
    )
    ax.set_title("Prompt Sensitivity Score by Language", fontweight="bold")
    ax.set_xlabel("Language")
    ax.set_ylabel("Prompt Sensitivity Score")

    for i, lang in enumerate(languages):
        subset = df.loc[df["language"] == lang, "prompt_sensitivity_score"].dropna()
        if subset.empty:
            ax.text(i, 0.01, "no data\n(null ECR)", ha="center", va="bottom",
                    fontsize=9, color="grey", style="italic")

    _save(fig, out)


# ---------------------------------------------------------------------------
# Plot 3 — Semantic Similarity vs Entity Change Rate (2×2 facet, jittered)
# ---------------------------------------------------------------------------

def plot_semantic_vs_entity(df: pd.DataFrame, out: str) -> None:
    """
    2×2 facet grid — one panel per language.

    Entity change rate is discrete (multiples of 0.25), so raw scatter
    produces dense horizontal bands.  A small Y-jitter spreads the points
    without misrepresenting the data.  Per-panel trend lines show whether
    semantic drift correlates with factual instability within each language.
    """
    plot_df = df.dropna(subset=["semantic_similarity", "entity_change_rate"]).copy()

    rng = np.random.default_rng(42)
    plot_df["ecr_jittered"] = (
        plot_df["entity_change_rate"]
        + rng.uniform(-0.03, 0.03, size=len(plot_df))
    )

    languages = sorted(plot_df["language"].unique())
    n_cols = 2
    n_rows = (len(languages) + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 8), sharex=False, sharey=False)
    axes_flat = axes.flatten()

    for i, lang in enumerate(languages):
        ax  = axes_flat[i]
        grp = plot_df[plot_df["language"] == lang]
        color = LANG_PALETTE.get(lang, "#888888")

        ax.scatter(
            grp["semantic_similarity"],
            grp["ecr_jittered"],
            color=color,
            alpha=0.45,
            s=22,
            edgecolors="none",
            rasterized=True,
        )

        # Trend line.
        if len(grp) > 2:
            x = grp["semantic_similarity"].values
            y = grp["entity_change_rate"].values      # use true ECR for fit
            m, b = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 200)
            ax.plot(x_line, m * x_line + b,
                    color="black", linewidth=1.4, linestyle="--",
                    label=f"slope {m:.2f}")
            ax.legend(fontsize=9, handlelength=1.2)

        # True ECR tick positions.
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.0"], fontsize=9)
        ax.set_title(f"Language: {lang}", fontsize=12, fontweight="bold", color=color)
        ax.set_xlabel("Semantic Similarity", fontsize=10)
        ax.set_ylabel("Entity Change Rate", fontsize=10)
        ax.grid(True, alpha=0.3)

    # Hide any unused subplots (if languages count is odd).
    for j in range(len(languages), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Semantic Similarity vs Entity Change Rate",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()

    _save(fig, out)


# ---------------------------------------------------------------------------
# Plot 4 — Top Unstable Prompts (most → least, large labels)
# ---------------------------------------------------------------------------

def plot_top_unstable(unstable: pd.DataFrame, out: str, top_n: int = 20) -> None:
    """
    Horizontal bar chart — most unstable prompt at the top, least at the bottom.

    Bars are sorted descending so the longest (most unstable) bar appears first.
    Font sizes are intentionally large for readability in reports.
    """
    data = unstable.head(top_n).copy()

    # Sort ascending so that when plotted horizontally the top bar = highest score.
    data = data.sort_values("avg_prompt_sensitivity_score", ascending=True)

    labels = [f"prompt {pid}" for pid in data["prompt_id"]]
    values = data["avg_prompt_sensitivity_score"].values

    norm   = plt.Normalize(values.min(), values.max())
    colors = plt.cm.YlOrRd(norm(values))  # type: ignore[attr-defined]

    fig_height = max(6, len(data) * 0.45)
    fig, ax = plt.subplots(figsize=(11, fig_height))

    bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.72)

    # Value labels — large font, offset slightly past bar end.
    offset = values.max() * 0.012
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title(f"Top {len(data)} Most Unstable Prompts  (most → least)",
                 fontweight="bold")
    ax.set_xlabel("Avg Prompt Sensitivity Score", fontsize=13)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.tick_params(axis="y", labelsize=11)
    ax.margins(x=0.18)

    _save(fig, out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate clean PSS plots (reads CSVs only, no model calls).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scores",   default="analysis/pss_scores.csv")
    parser.add_argument("--unstable", default="analysis/unstable_prompts.csv")
    parser.add_argument("--out_dist", default="plots/pss_distribution.png")
    parser.add_argument("--out_lang", default="plots/pss_by_language.png")
    parser.add_argument("--out_scat", default="plots/semantic_vs_entity.png")
    parser.add_argument("--out_top",  default="plots/top_unstable_prompts.png")
    parser.add_argument("--top_n",    type=int, default=20,
                        help="Number of most-unstable prompts to show.")
    return parser.parse_args()


def _setup_file_logging() -> None:
    from datetime import datetime
    from pathlib import Path as _Path
    _Path("logs").mkdir(exist_ok=True)
    fh = logging.FileHandler(
        f"logs/plot_pss_{datetime.now().strftime('%Y%m%d')}.log", encoding="utf-8"
    )
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(fh)


def main() -> None:
    args = _parse_args()
    _setup_file_logging()
    logger.info("Starting plot stage — plot_pss_clean.py")

    # 1. Clean slate — remove stale plots.
    delete_old_plots()

    # 2. Load data.
    scores_df   = pd.read_csv(args.scores)
    unstable_df = pd.read_csv(args.unstable)
    logger.info("Loaded %d score rows, %d unstable-prompt rows.",
                len(scores_df), len(unstable_df))

    # 3. Generate plots.
    plot_pss_distribution(scores_df,   args.out_dist)
    plot_pss_by_language(scores_df,    args.out_lang)
    plot_semantic_vs_entity(scores_df, args.out_scat)
    plot_top_unstable(unstable_df,     args.out_top, top_n=args.top_n)

    logger.info("All plots saved to %s/", PLOTS_DIR)


if __name__ == "__main__":
    main()

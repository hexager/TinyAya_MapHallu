import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid", context="talk")


def normalize_pairs(df: pd.DataFrame, english_code: str = "en") -> pd.DataFrame:
    """Normalize pair labels (en first if present, otherwise alphabetical)."""
    df = df.copy()

    def make_pair(row):
        a, b = row["lang_a"], row["lang_b"]
        if a == english_code or b == english_code:
            other = b if a == english_code else a
            return f"{english_code}-{other}"
        x, y = sorted([a, b])
        return f"{x}-{y}"

    df["pair"] = df.apply(make_pair, axis=1)
    return df


def compute_pair_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["pair", "lang_a", "lang_b"], as_index=False)
        .agg(
            n=("pair", "size"),
            label_disagreement_mean=("label_disagreement", "mean"),
            label_disagreement_std=("label_disagreement", "std"),
            confidence_distance_mean=("confidence_distance", "mean"),
            confidence_distance_std=("confidence_distance", "std"),
        )
        .sort_values("label_disagreement_mean", ascending=False)
    )


def plot_pairwise_bars(summary_df: pd.DataFrame, out_dir: Path):
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=summary_df, x="pair", y="label_disagreement_mean", color="#4C78A8")
    ax.set_title("Mean Label Disagreement by Language Pair")
    ax.set_xlabel("Language Pair")
    ax.set_ylabel("Mean Label Disagreement (0-1)")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "cmdr_label_disagreement_bar.png", dpi=200)
    plt.close()

    sorted_conf = summary_df.sort_values("confidence_distance_mean", ascending=False)
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=sorted_conf, x="pair", y="confidence_distance_mean", color="#F58518")
    ax.set_title("Mean Confidence Distance by Language Pair")
    ax.set_xlabel("Language Pair")
    ax.set_ylabel("Mean |prob_a - prob_b|")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "cmdr_confidence_distance_bar.png", dpi=200)
    plt.close()


def plot_confidence_distribution(df: pd.DataFrame, out_dir: Path):
    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(data=df, x="pair", y="confidence_distance", color="#72B7B2")
    ax.set_title("Confidence Distance Distribution by Language Pair")
    ax.set_xlabel("Language Pair")
    ax.set_ylabel("|prob_a - prob_b|")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "cmdr_confidence_distance_box.png", dpi=200)
    plt.close()


def plot_pairwise_heatmap(df: pd.DataFrame, out_dir: Path):
    langs = sorted(set(df["lang_a"]).union(set(df["lang_b"])))
    pair_avg = df.groupby(["lang_a", "lang_b"], as_index=False)["label_disagreement"].mean()

    mat = pd.DataFrame(np.nan, index=langs, columns=langs)
    for _, row in pair_avg.iterrows():
        a, b = row["lang_a"], row["lang_b"]
        mat.loc[a, b] = row["label_disagreement"]
        mat.loc[b, a] = row["label_disagreement"]
    for lang in langs:
        mat.loc[lang, lang] = 0.0

    plt.figure(figsize=(8, 7))
    ax = sns.heatmap(
        mat.astype(float),
        annot=True,
        fmt=".2f",
        cmap="Reds",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"label": "Mean Label Disagreement"},
    )
    ax.set_title("Pairwise Label Disagreement Heatmap")
    ax.set_xlabel("Language")
    ax.set_ylabel("Language")
    plt.tight_layout()
    plt.savefig(out_dir / "cmdr_label_disagreement_heatmap.png", dpi=200)
    plt.close()


def plot_english_pairs(df: pd.DataFrame, out_dir: Path, english_code="en") -> pd.DataFrame:
    en_df = df[(df["lang_a"] == english_code) | (df["lang_b"] == english_code)].copy()
    if en_df.empty:
        return pd.DataFrame()

    en_df["other_lang"] = en_df.apply(
        lambda row: row["lang_b"] if row["lang_a"] == english_code else row["lang_a"], axis=1
    )
    en_agg = (
        en_df.groupby("other_lang", as_index=False)
        .agg(
            n=("other_lang", "size"),
            label_disagreement_mean=("label_disagreement", "mean"),
            confidence_distance_mean=("confidence_distance", "mean"),
        )
        .sort_values("label_disagreement_mean", ascending=False)
    )

    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=en_agg, x="other_lang", y="label_disagreement_mean", color="#E45756")
    ax.set_title("English vs Other Languages: Label Disagreement")
    ax.set_xlabel("Other Language")
    ax.set_ylabel("Mean Label Disagreement")
    plt.tight_layout()
    plt.savefig(out_dir / "cmdr_english_pairs_label_disagreement.png", dpi=200)
    plt.close()

    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=en_agg, x="other_lang", y="confidence_distance_mean", color="#54A24B")
    ax.set_title("English vs Other Languages: Confidence Distance")
    ax.set_xlabel("Other Language")
    ax.set_ylabel("Mean |prob_en - prob_other|")
    plt.tight_layout()
    plt.savefig(out_dir / "cmdr_english_pairs_confidence_distance.png", dpi=200)
    plt.close()

    return en_agg


def main():
    parser = argparse.ArgumentParser(description="Plot CMDR hallucination/stability metrics.")
    parser.add_argument(
        "--input-csv",
        default="outputs/cmdr_sample_metrics.csv",
        help="Path to cmdr sample metrics CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/plots",
        help="Directory where plot PNGs and summary CSVs will be written.",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    df = normalize_pairs(df, english_code="en")

    pair_summary = compute_pair_summary(df)
    pair_summary.to_csv(out_dir / "cmdr_pair_summary.csv", index=False)

    en_summary = plot_english_pairs(df, out_dir, english_code="en")
    if not en_summary.empty:
        en_summary.to_csv(out_dir / "cmdr_english_summary.csv", index=False)

    plot_pairwise_bars(pair_summary, out_dir)
    plot_confidence_distribution(df, out_dir)
    plot_pairwise_heatmap(df, out_dir)

    print(f"Plots and summaries written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
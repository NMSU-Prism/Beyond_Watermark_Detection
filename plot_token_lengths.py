from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

parser = argparse.ArgumentParser(
    description="Save individual final-text token-length plots."
)
parser.add_argument("--runs-dir", default="runs")
parser.add_argument("--expected-n", type=int, default=6000)
args = parser.parse_args()

runs = Path(args.runs_dir).expanduser().resolve()
out = runs / "comparison_all_methods/token_length_plots/individual"

experiments = {
    "KGW–KGW": {
        "slug": "KGW_KGW",
        "folder": runs / "window_sweep_KGW",
        "color": "#0072B2",
    },
    "PRW–PRW": {
        "slug": "PRW_PRW",
        "folder": runs / "window_sweep_PRW_PRW_actualmin25_average",
        "color": "#D55E00",
    },
    "KGW–PRW": {
        "slug": "KGW_PRW",
        "folder": runs / "window_sweep_KGW_PRW",
        "color": "#009E73",
    },
}

def read_lengths(method, folder):
    # Read one layout only: each layout contains the same passages.
    candidates = [
        folder / "dynamic4/features.csv",
        folder / "whole/features.csv",
        folder / "features.csv",
    ]
    source = next((p for p in candidates if p.exists()), None)

    if source is None:
        raise SystemExit(
            f"{method}: no features.csv found in {folder}"
        )

    df = pd.read_csv(source, dtype={"idx": str})

    if {"A_n_tokens", "B_n_tokens"}.issubset(df.columns):
        a = pd.to_numeric(df["A_n_tokens"], errors="raise")
        b = pd.to_numeric(df["B_n_tokens"], errors="raise")

        if not np.array_equal(a.to_numpy(), b.to_numpy()):
            raise SystemExit(
                f"{method}: A/B token counts differ. "
                "Select a common tokenizer before comparing lengths."
            )
        lengths = a

    elif "n_tokens" in df.columns:
        lengths = pd.to_numeric(df["n_tokens"], errors="raise")

    else:
        raise SystemExit(
            f"{source}: no supported token-count column.\n"
            f"Available columns: {list(df.columns)}"
        )

    if len(df) != args.expected_n:
        raise SystemExit(
            f"{method}: expected {args.expected_n} passages, "
            f"found {len(df)}."
        )

    if (
        lengths.isna().any()
        or not np.isfinite(lengths.to_numpy()).all()
        or (lengths <= 0).any()
        or not np.equal(lengths, np.floor(lengths)).all()
    ):
        raise SystemExit(f"{method}: invalid token counts.")

    if "sample_id" in df.columns:
        if (
            df["sample_id"].isna().any()
            or df["sample_id"].duplicated().any()
        ):
            raise SystemExit(
                f"{method}: missing or duplicate sample IDs."
            )
    elif {"idx", "label"}.issubset(df.columns):
        if df.duplicated(["idx", "label"]).any():
            raise SystemExit(
                f"{method}: duplicate idx/label pairs."
            )

    print(f"{method}: {len(df):,} passages from {source}")
    return lengths.astype(int).reset_index(drop=True), source

# Validate all inputs before writing output.
datasets = {}
for method, settings in experiments.items():
    lengths, source = read_lengths(method, settings["folder"])
    datasets[method] = {
        **settings,
        "lengths": lengths,
        "source": source,
    }

out.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 21,
    "axes.labelsize": 20,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 16,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Shared axes allow comparisons across separate figures.
maximum_length = max(
    item["lengths"].max() for item in datasets.values()
)

def save(fig, filename):
    for extension in ("png", "pdf"):
        path = out / f"{filename}.{extension}"
        fig.savefig(path, dpi=400, bbox_inches="tight")
        print("Saved:", path)
    plt.close(fig)

summaries = []

for method, item in datasets.items():
    lengths = item["lengths"]
    slug = item["slug"]
    color = item["color"]

    # Save lengths for later reuse.
    pd.DataFrame({"n_tokens": lengths}).to_csv(
        out / f"{slug}_token_lengths.csv",
        index=False,
    )

    summary = lengths.describe(
        percentiles=[0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    )
    summary.name = method
    summaries.append(summary)

    summary.to_csv(
        out / f"{slug}_token_length_summary.csv",
        header=["tokens"],
    )

    # Figure 1: individual empirical CDF.
    # At x tokens, F(x) is the fraction of passages with length <= x.
    x, counts = np.unique(lengths, return_counts=True)
    cumulative = np.cumsum(counts) / counts.sum()

    pd.DataFrame({
        "n_tokens": x,
        "cumulative_fraction": cumulative,
    }).to_csv(
        out / f"{slug}_token_length_cdf_points.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Start at zero so the first jump is visible.
    ax.step(
        np.r_[0, x],
        np.r_[0, cumulative],
        where="post",
        color=color,
        linewidth=3,
        label=f"Final passages (n={len(lengths):,})",
    )
    ax.set_xlabel("Final-passage length (tokens)")
    ax.set_ylabel("Cumulative proportion")
    ax.set_title(f"{method}: token-length CDF")
    ax.set_xlim(0, maximum_length * 1.03)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc="lower right", frameon=True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    save(fig, f"{slug}_token_length_cdf")

    # Figure 2: individual box plot, without history categories.
    fig, ax = plt.subplots(figsize=(6, 5.5))

    sns.boxplot(
        y=lengths,
        color=color,
        width=0.4,
        showmeans=True,
        fliersize=3,
        linewidth=1.6,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 8,
        },
        ax=ax,
    )

    ax.set_title(f"{method}: token lengths")
    ax.set_ylabel("Final-passage length (tokens)")
    ax.set_xlabel(f"All {len(lengths):,} final passages")
    ax.set_xticks([])
    ax.set_ylim(0, maximum_length * 1.08)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    save(fig, f"{slug}_token_length_boxplot")

    print(f"\n{method} statistics:")
    print(summary.round(2).to_string())
    print()


# ============================================================
# COMBINED FIGURE 1: CDF of all three methods
# ============================================================

fig, ax = plt.subplots(figsize=(10, 7))

for method, item in datasets.items():
    lengths = item["lengths"]
    color = item["color"]

    x, counts = np.unique(lengths, return_counts=True)
    cumulative = np.cumsum(counts) / counts.sum()

    ax.step(
        np.r_[0, x],
        np.r_[0, cumulative],
        where="post",
        color=color,
        linewidth=3,
        label=method,
    )

ax.set_xlabel("Final-passage length (tokens)")
ax.set_ylabel("Cumulative proportion")
ax.set_title("Final-passage token-length distributions")
ax.set_xlim(0, maximum_length * 1.03)
ax.set_ylim(0, 1.02)
ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.legend(loc="lower right", frameon=True)
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
save(fig, "ALL_METHODS_token_length_cdf")


# ============================================================
# COMBINED FIGURE 2: Boxplot of all three methods
# ============================================================

fig, ax = plt.subplots(figsize=(9, 7))

plot_data = [
    item["lengths"].to_numpy()
    for item in datasets.values()
]

plot_labels = list(datasets.keys())
plot_colors = [
    item["color"]
    for item in datasets.values()
]

box = ax.boxplot(
    plot_data,
    tick_labels=plot_labels,
    patch_artist=True,
    showmeans=True,
    showfliers=True,
    widths=0.5,
    meanprops={
        "marker": "D",
        "markerfacecolor": "white",
        "markeredgecolor": "black",
        "markersize": 8,
    },
    medianprops={
        "color": "black",
        "linewidth": 2,
    },
)

for patch, color in zip(box["boxes"], plot_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

ax.set_ylabel("Final-passage length (tokens)")
ax.set_title("Token-length comparison across methods")
ax.set_ylim(0, maximum_length * 1.08)
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
save(fig, "ALL_METHODS_token_length_boxplot")

print("\nCombined CDF and boxplot generated successfully.")

# Also save one summary CSV for convenient comparison.
summary_table = pd.DataFrame(summaries)
summary_table.index.name = "method_pair"
summary_table.to_csv(out / "token_length_summary_all_methods.csv")

print("Token-length statistics:")
print(summary_table.round(2).to_string())
print("\nSaved individual plots and statistics in:", out)
print("No generation, scoring, or classifier training was repeated.")

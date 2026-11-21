from pathlib import Path
import argparse

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root",
    default=str(Path.home() / "order_detect/runs/windows_only_three_pairs_1buCwy/KGW_PRW")
)
parser.add_argument("--title", default="KGW–PRW")
args = parser.parse_args()

root = Path(args.root).expanduser()
df = pd.read_csv(root / "all_results.csv")

out = root / "comparison_plots"
out.mkdir(parents=True, exist_ok=True)

# Larger text and vector PDF fonts for paper figures.
plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "figure.titlesize": 21,
    "lines.linewidth": 2.6,
    "lines.markersize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

models = {
    "linear": ("Linear regression", "#0072B2", "o"),
    "svm": ("RBF SVM", "#D55E00", "s"),
    "mlp": ("Neural network", "#009E73", "^"),
}

dynamic_counts = list(range(4, 9))
fixed25_counts = list(range(4, 9))

layouts = (
    ["whole"]
    + [f"dynamic{k}" for k in dynamic_counts]
    + [f"fixed25x{k}" for k in fixed25_counts]
)

labels = {
    "whole": "Whole text only",
    **{f"dynamic{k}": f"Dynamic {k} sections" for k in dynamic_counts},
    **{
        f"fixed25x{k}": f"{k} × 25-token windows"
        for k in fixed25_counts
    },
}

required = {"split", "model", "layout", "accuracy"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

selected = df[
    df["split"].isin(["validation", "test"])
    & df["layout"].isin(layouts)
    & df["model"].isin(models)
].copy()

if selected.duplicated(["split", "layout", "model"]).any():
    raise ValueError("Duplicate split/layout/model results found.")

for split in ("test",):
    for model in models:
        available = set(selected.loc[
            selected["split"].eq(split)
            & selected["model"].eq(model), "layout"
        ])
        absent = set(layouts) - available
        if absent:
            raise ValueError(
                f"{split}/{model}: missing layouts {sorted(absent)}"
            )

def save(fig, name):
    for extension in ("png", "pdf"):
        path = out / f"{name}.{extension}"
        fig.savefig(path, dpi=400, bbox_inches="tight")
        print("Saved:", path)
    plt.close(fig)

legend_handles = [
    Line2D(
        [0], [0], color=color, marker=marker,
        linewidth=2.6, markersize=8, label=name
    )
    for name, color, marker in models.values()
]

# Figure 1: all retained layouts, including whole text.
fig, axes = plt.subplots(
    1, 2, figsize=(16, 10), sharex=True, sharey=True
)

offsets = {"linear": -0.22, "svm": 0, "mlp": 0.22}

for ax, split in zip(axes, ("test",)):
    subset = selected[selected["split"].eq(split)]

    for model, (name, color, marker) in models.items():
        values = (
            subset[subset["model"].eq(model)]
            .set_index("layout")["accuracy"]
            .reindex(layouts)
        )
        positions = [i + offsets[model] for i in range(len(layouts))]
        ax.scatter(
            values, positions, label=name, color=color,
            marker=marker, s=95, edgecolors="white",
            linewidths=0.6, zorder=3
        )

    # ax.set_title(split.capitalize(), pad=14)
    ax.set_xlabel("Accuracy", labelpad=10)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # Separate whole text, dynamic sections and fixed windows.
    ax.axhline(0.5, color="gray", alpha=0.3)
    ax.axhline(6.5, color="gray", alpha=0.3)

axes[0].set_yticks(range(len(layouts)))
axes[0].set_yticklabels([labels[x] for x in layouts])
axes[0].invert_yaxis()

fig.suptitle(
    f"{args.title}: feature-layout comparison",
    y=0.98
)
fig.legend(
    handles=legend_handles,
    loc="lower center", bbox_to_anchor=(0.5, 0.01),
    ncol=3, frameon=False
)
fig.tight_layout(rect=(0, 0.09, 1, 0.94))
save(fig, "all_layouts_accuracy")

# Figure 2: two window families; no fixed 59-token panels.
fig, axes = plt.subplots(
    2, 2, figsize=(15, 10),
    sharex=False, sharey=True
)

families = [
    ("dynamic", "Dynamic sections", dynamic_counts),
    ("fixed25x", "Fixed 25-token windows", fixed25_counts),
]

for row, split in enumerate(("test",)):
    subset = selected[selected["split"].eq(split)]

    for col, (prefix, title, counts) in enumerate(families):
        ax = axes[row, col]

        for model, (name, color, marker) in models.items():
            scores = (
                subset[subset["model"].eq(model)]
                .set_index("layout")["accuracy"]
            )
            values = [scores.loc[f"{prefix}{k}"] for k in counts]

            ax.plot(
                counts, values,
                marker=marker, color=color, label=name,
                zorder=3
            )
            ax.axhline(
                scores.loc["whole"],
                color=color, linestyle="--",
                linewidth=2, alpha=0.6
            )

        # ax.set_title(
        #     f"{split.capitalize()}: {title}", pad=12
        # )
        ax.set_xticks(counts)
        ax.set_xlim(min(counts) - 0.2, max(counts) + 0.2)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

        if row == 1:
            ax.set_xlabel("Number of windows", labelpad=10)
        if col == 0:
            ax.set_ylabel("Accuracy", labelpad=10)

fig.suptitle(
    f"{args.title}: effect of window count",
    y=0.99
)
fig.legend(
    handles=legend_handles,
    loc="lower center", bbox_to_anchor=(0.5, 0.035),
    ncol=3, frameon=False
)
fig.text(
    0.5, 0.01,
    "Dashed lines: corresponding whole-text-only baselines.",
    ha="center", fontsize=15
)
fig.tight_layout(rect=(0, 0.13, 1, 0.95))
save(fig, "window_count_trends")

# Figure 3: overlay dynamic and fixed-25 results for each split.
# fig, axes = plt.subplots(
#     1, 2, figsize=(17, 7), sharey=True
# )

# for ax, split in zip(axes, ("test",)):
fig, ax = plt.subplots(
    1, 1, figsize=(9, 7)
)

for split in ("test",):
    subset = selected[selected["split"].eq(split)]

    for model, (name, color, marker) in models.items():
        scores = (
            subset[subset["model"].eq(model)]
            .set_index("layout")["accuracy"]
        )

        # Dynamic sections: 4 to 9.
        ax.plot(
            dynamic_counts,
            [scores.loc[f"dynamic{k}"] for k in dynamic_counts],
            color=color,
            marker=marker,
            linestyle="-",
            linewidth=2.8,
            markersize=9,
            markerfacecolor=color,
            markeredgecolor=color,
            zorder=3,
        )

        # Fixed 25-token windows: 4 to 8.
        ax.plot(
            fixed25_counts,
            [scores.loc[f"fixed25x{k}"] for k in fixed25_counts],
            color=color,
            marker=marker,
            linestyle="--",
            linewidth=2.8,
            markersize=10,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=2,
            zorder=4,
        )

        # Whole-text-only baseline for this classifier.
        ax.axhline(
            scores.loc["whole"],
            color=color,
            linestyle=":",
            linewidth=2,
            alpha=0.7,
            zorder=2,
        )

    # ax.set_title(split.capitalize(), pad=14)
    ax.set_xlabel("Number of windows", labelpad=10)
    ax.set_xticks(dynamic_counts)
    ax.set_xlim(3.8, 9.2)
    ax.yaxis.set_major_formatter(
        PercentFormatter(1, decimals=1)
    )
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

ax.set_ylabel("Accuracy", labelpad=10)

# First legend: classifier colors.
classifier_handles = [
    Line2D(
        [0], [0],
        color=color,
        marker=marker,
        linestyle="-",
        linewidth=2.8,
        markersize=9,
        label=name,
    )
    for name, color, marker in models.values()
]

# Second legend: window-family line styles.
layout_handles = [
    Line2D(
        [0], [0], color="black",
        marker="o", linestyle="-",
        linewidth=2.8, markersize=9,
        label="Dynamic sections",
    ),
    Line2D(
        [0], [0], color="black",
        marker="o", linestyle="--",
        markerfacecolor="white",
        markeredgewidth=2,
        linewidth=2.8, markersize=9,
        label="Fixed 25-token windows",
    ),
    Line2D(
        [0], [0], color="black",
        linestyle=":", linewidth=2,
        label="Whole-text baseline",
    ),
]

# fig.suptitle(
#     f"{args.title}: dynamic vs. fixed 25-token windows",
#     y=0.98,
# )

fig.legend(
    handles=classifier_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.075),
    ncol=3,
    frameon=False,
    fontsize=18,
)

fig.legend(
    handles=layout_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.005),
    ncol=3,
    frameon=False,
    fontsize=17,
)

fig.tight_layout(rect=(0, 0.22, 1, 0.92))
save(fig, "dynamic_vs_fixed25_validation_test")

print("\nFinished. No generation or training repeated.")

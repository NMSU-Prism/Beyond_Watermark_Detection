from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.lines import Line2D


parser = argparse.ArgumentParser(
    description="Plot test results for whole text, dynamic and fixed-25 windows."
)
parser.add_argument(
    "--root",
    default=str(
        Path.home()
        / "order_detect/runs/windows_only_three_pairs_1buCwy/KGW_KGW"
    ),
)
parser.add_argument("--title", default="KGW–KGW")
parser.add_argument("--min-windows", type=int, default=4)
parser.add_argument("--max-windows", type=int, default=8)
args = parser.parse_args()

if not 1 <= args.min_windows <= args.max_windows:
    parser.error("Require 1 <= min-windows <= max-windows")

root = Path(args.root).expanduser()
source = root / "all_results.csv"

# Separate directory preserves your earlier figures.
out = root / "comparison_plots_with_whole_test"

df = pd.read_csv(source)

plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 17,
    "figure.titlesize": 21,
    "lines.linewidth": 2.8,
    "lines.markersize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

models = {
    "linear": ("Linear regression", "#0072B2", "o"),
    "svm": ("RBF SVM", "#D55E00", "s"),
    "mlp": ("Neural network", "#009E73", "^"),
}

counts = list(range(args.min_windows, args.max_windows + 1))

layouts = (
    ["whole"]
    + [f"dynamic{k}" for k in counts]
    + [f"fixed25x{k}" for k in counts]
)

labels = {
    "whole": "Whole text only",
    **{f"dynamic{k}": f"Dynamic {k} sections" for k in counts},
    **{
        f"fixed25x{k}": f"{k} × 25-token windows"
        for k in counts
    },
}

# Validate all requested data before saving plots.
required = {"split", "model", "layout", "accuracy"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"{source}: missing columns {sorted(missing)}")

for column in ("split", "model", "layout"):
    df[column] = df[column].astype(str).str.strip()

df["split"] = df["split"].str.lower()

selected = df[
    df["split"].eq("test")
    & df["layout"].isin(layouts)
    & df["model"].isin(models)
].copy()

if selected.duplicated(["layout", "model"]).any():
    raise ValueError("Duplicate test/layout/model results found.")

selected["accuracy"] = pd.to_numeric(
    selected["accuracy"], errors="raise"
)

if (
    not np.isfinite(selected["accuracy"].to_numpy()).all()
    or not selected["accuracy"].between(0, 1).all()
):
    raise ValueError("Accuracy must be finite and between 0 and 1.")

for model in models:
    available = set(
        selected.loc[selected["model"].eq(model), "layout"]
    )
    absent = set(layouts) - available

    if absent:
        raise ValueError(
            f"test/{model}: missing layouts {sorted(absent)}. "
            "Check --root and the requested window range."
        )

out.mkdir(parents=True, exist_ok=True)


def scores_for(model):
    return (
        selected[selected["model"].eq(model)]
        .set_index("layout")["accuracy"]
    )


def save(fig, name):
    for extension in ("png", "pdf"):
        path = out / f"{name}.{extension}"
        fig.savefig(path, dpi=400, bbox_inches="tight")
        print("Saved:", path)
    plt.close(fig)


def format_window_axis(ax):
    ax.set_xlabel("Number of windows", labelpad=10)
    ax.set_xticks(counts)
    ax.set_xlim(counts[0] - 0.2, counts[-1] + 0.2)
    ax.yaxis.set_major_formatter(
        PercentFormatter(1, decimals=1)
    )
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


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

layout_handles = [
    Line2D(
        [0], [0],
        color="black",
        marker="o",
        linestyle="-",
        label="Dynamic sections",
    ),
    Line2D(
        [0], [0],
        color="black",
        marker="o",
        linestyle="--",
        markerfacecolor="white",
        markeredgewidth=2,
        label="Fixed 25-token windows",
    ),
    Line2D(
        [0], [0],
        color="black",
        marker="o",
        linestyle=":",
        label="Whole-text baseline",
    ),
]


# ------------------------------------------------------------
# Figure 1: all layouts, including the whole-text-only result.
# ------------------------------------------------------------
fig, ax = plt.subplots(
    figsize=(12, max(8, 0.5 * len(layouts) + 2))
)

offsets = {"linear": -0.22, "svm": 0, "mlp": 0.22}

for model, (name, color, marker) in models.items():
    values = scores_for(model).reindex(layouts)
    positions = np.arange(len(layouts)) + offsets[model]

    ax.scatter(
        values,
        positions,
        color=color,
        marker=marker,
        s=100,
        edgecolors="white",
        linewidths=0.6,
        zorder=3,
    )

ax.set_yticks(range(len(layouts)))
ax.set_yticklabels([labels[layout] for layout in layouts])
ax.invert_yaxis()

# Separate whole text, dynamic sections, and fixed windows.
ax.axhline(0.5, color="gray", alpha=0.35)
ax.axhline(
    len(counts) + 0.5,
    color="gray",
    alpha=0.35,
)

ax.set_xlabel("Test accuracy", labelpad=10)
ax.xaxis.set_major_formatter(
    PercentFormatter(1, decimals=1)
)
ax.grid(axis="x", alpha=0.25)
ax.spines[["top", "right"]].set_visible(False)

# ax.set_title(
#     f"{args.title}: feature-layout comparison",
#     pad=16,
# )

fig.legend(
    handles=classifier_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.01),
    ncol=3,
    frameon=False,
)

fig.tight_layout(rect=(0, 0.09, 1, 1))
save(fig, "test_all_layouts_accuracy_with_whole")


# ------------------------------------------------------------
# Figure 2: separate dynamic and fixed-25 panels.
# Dotted lines repeat the whole-text baseline across the counts.
# ------------------------------------------------------------
fig, axes = plt.subplots(
    1, 2,
    figsize=(15, 7),
    sharey=True,
)

families = [
    ("dynamic", "Dynamic sections"),
    ("fixed25x", "Fixed 25-token windows"),
]

for ax, (prefix, title) in zip(axes, families):
    for model, (name, color, marker) in models.items():
        scores = scores_for(model)

        ax.plot(
            counts,
            [scores.loc[f"{prefix}{k}"] for k in counts],
            color=color,
            marker=marker,
            linestyle="-",
            zorder=3,
        )

        ax.plot(
            counts,
            [scores.loc["whole"]] * len(counts),
            color=color,
            marker=marker,
            linestyle=":",
            linewidth=2,
            markersize=7,
            alpha=0.8,
            zorder=2,
        )

    ax.set_title(title, pad=14)
    format_window_axis(ax)

axes[0].set_ylabel("Test accuracy", labelpad=10)

# fig.suptitle(
#     f"{args.title}: effect of window count",
#     y=0.99,
# )

fig.legend(
    handles=classifier_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.055),
    ncol=3,
    frameon=False,
)

fig.text(
    0.5, 0.015,
    "Solid: window-based results. Dotted: whole-text-only baselines.",
    ha="center",
    fontsize=15,
)

fig.tight_layout(rect=(0, 0.17, 1, 0.93))
save(fig, "test_window_count_trends_with_whole")


# ------------------------------------------------------------
# Figure 3: dynamic, fixed-25, and whole baselines together.
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 8))

for model, (name, color, marker) in models.items():
    scores = scores_for(model)

    # Dynamic windows: solid lines with filled markers.
    ax.plot(
        counts,
        [scores.loc[f"dynamic{k}"] for k in counts],
        color=color,
        marker=marker,
        linestyle="-",
        markerfacecolor=color,
        markeredgecolor=color,
        zorder=3,
    )

    # Fixed 25-token windows: dashed lines with hollow markers.
    ax.plot(
        counts,
        [scores.loc[f"fixed25x{k}"] for k in counts],
        color=color,
        marker=marker,
        linestyle="--",
        markersize=10,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=2,
        zorder=4,
    )

    # Whole text: one baseline repeated across all displayed counts.
    ax.plot(
        counts,
        [scores.loc["whole"]] * len(counts),
        color=color,
        marker=marker,
        linestyle=":",
        linewidth=2,
        markersize=7,
        alpha=0.8,
        zorder=2,
    )

ax.set_ylabel("Test accuracy", labelpad=10)
# ax.set_title(
#     f"{args.title}: dynamic, fixed-25 and whole text",
#     pad=16,
# )
format_window_axis(ax)

fig.legend(
    handles=classifier_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.08),
    ncol=3,
    frameon=False,
)

fig.legend(
    handles=layout_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.015),
    ncol=3,
    frameon=False,
    fontsize=15,
)

fig.tight_layout(rect=(0, 0.21, 1, 1))
save(fig, "test_dynamic_fixed25_whole_overlay")

# Save the exact results used in the figures.
selected.to_csv(
    out / "plotted_test_results.csv",
    index=False,
)

print("\nFinished. No generation, scoring, or training repeated.")
print("Whole-text values are repeated baselines, not separate experiments.")

from pathlib import Path
import argparse

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root",
    default=str(Path.home() / "order_detect/runs/window_sweep_KGW")
)
args = parser.parse_args()

root = Path(args.root).expanduser()
df = pd.read_csv(root / "all_results.csv")

out = root / "comparison_plots"
out.mkdir(exist_ok=True)

models = {
    "linear": ("Linear regression", "#0072B2"),
    "svm": ("RBF SVM", "#D55E00"),
    "mlp": ("Neural network", "#009E73"),
}

layouts = (
    ["whole"]
    + [f"dynamic{k}" for k in range(4, 10)]
    + [f"fixed25x{k}" for k in range(4, 10)]
    + [f"fixed59x{k}" for k in range(4, 10)]
)

labels = {
    "whole": "Whole text only",
    **{f"dynamic{k}": f"Dynamic {k} sections" for k in range(4, 10)},
    **{f"fixed25x{k}": f"{k} × 25-token windows" for k in range(4, 10)},
    **{f"fixed59x{k}": f"{k} × 59-token windows" for k in range(4, 10)},
}

# Figure 1: compare every layout, including whole-text baseline.
fig, axes = plt.subplots(
    1, 2, figsize=(13, 11), sharex=True, sharey=True
)

offsets = {"linear": -0.22, "svm": 0, "mlp": 0.22}

for ax, split in zip(axes, ["validation", "test"]):
    subset = df[df["split"] == split]

    for model, (name, color) in models.items():
        values = (
            subset[subset["model"] == model]
            .set_index("layout")["accuracy"]
            .reindex(layouts)
        )
        positions = [i + offsets[model] for i in range(len(layouts))]
        ax.scatter(values, positions, label=name, color=color, s=38)

    ax.set_title(split.capitalize())
    ax.set_xlabel("Accuracy")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="x", alpha=0.25)

axes[0].set_yticks(range(len(layouts)))
axes[0].set_yticklabels([labels[x] for x in layouts])
axes[0].invert_yaxis()
axes[1].legend(loc="lower right")

fig.suptitle("Watermark-order classification: feature comparison", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.96])

for extension in ("png", "pdf"):
    fig.savefig(out / f"all_layouts_accuracy.{extension}", dpi=300)
plt.close(fig)

# Figure 2: window-count trends. Dashed lines show each model's
# whole-text-only baseline; all window layouts also include whole scores.
fig, axes = plt.subplots(
    2, 3, figsize=(15, 8), sharex=True, sharey=True
)

families = [
    ("dynamic", "Dynamic sections"),
    ("fixed25x", "Fixed 25-token windows"),
    ("fixed59x", "Fixed 59-token windows"),
]

for row, split in enumerate(["validation", "test"]):
    subset = df[df["split"] == split]

    for col, (prefix, title) in enumerate(families):
        ax = axes[row, col]

        for model, (name, color) in models.items():
            scores = subset[subset["model"] == model].set_index("layout")
            values = [
                scores.loc[f"{prefix}{k}", "accuracy"]
                for k in range(4, 10)
            ]
            ax.plot(
                range(4, 10), values,
                marker="o", color=color, label=name
            )
            ax.axhline(
                scores.loc["whole", "accuracy"],
                color=color, linestyle="--", alpha=0.6
            )

        ax.set_title(f"{split.capitalize()}: {title}")
        ax.set_xticks(range(4, 10))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=0.25)

        if row == 1:
            ax.set_xlabel("Number of windows")
        if col == 0:
            ax.set_ylabel("Accuracy")

axes[0, 0].legend(fontsize=9)
fig.suptitle(
    "Window-count effects — dashed lines: whole-text-only baselines",
    fontsize=14
)
fig.tight_layout(rect=[0, 0, 1, 0.95])

for extension in ("png", "pdf"):
    fig.savefig(out / f"window_count_trends.{extension}", dpi=300)
plt.close(fig)

print("Saved plots in:", out)

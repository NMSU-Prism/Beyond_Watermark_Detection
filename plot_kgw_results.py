from pathlib import Path
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
source = root / "runs/window_sweep_KGW/all_results.csv"
out = root / "runs/window_sweep_KGW/presentation_plots"
out.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(source)
df = df[df["split"].eq("test")].copy()

required = {
    "layout", "windows", "model", "accuracy", "macro_f1", "auc"
}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

if df.duplicated(["layout", "model"]).any():
    raise ValueError("Duplicate layout/model rows found.")

models = {
    "linear": ("Linear regression", "#0072B2", "o"),
    "svm": ("SVM", "#D55E00", "s"),
    "mlp": ("Neural network", "#009E73", "^"),
}

families = {
    "Dynamic windows": "dynamic",
    "Fixed windows: 25 tokens": "fixed25x",
    "Fixed windows: 59 tokens": "fixed59x",
}

metrics = {
    "accuracy": ("Test accuracy (%)", 100),
    "macro_f1": ("Test macro F1", 1),
    "auc": ("Test ROC-AUC", 1),
}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

for metric, (ylabel, scale) in metrics.items():
    fig, axes = plt.subplots(
        1, 3, figsize=(13, 4.4), sharey=True
    )

    for ax, (title, prefix) in zip(axes, families.items()):
        family = df[
            df["layout"].str.startswith(prefix)
        ].copy()

        for model, (name, color, marker) in models.items():
            rows = family[family["model"].eq(model)].sort_values("windows")
            baseline = df[
                df["layout"].eq("whole") & df["model"].eq(model)
            ]

            if rows.empty or len(baseline) != 1:
                raise ValueError(f"Missing results: {title}, {model}")

            ax.plot(
                rows["windows"], scale * rows[metric],
                color=color, marker=marker,
                linewidth=2, markersize=5,
                label=name,
            )

            ax.axhline(
                scale * baseline.iloc[0][metric],
                color=color, linestyle="--",
                linewidth=1.3, alpha=0.65,
            )

        ax.set_title(title)
        ax.set_xlabel("Number of windows")
        ax.set_xticks([4, 5, 6, 7, 8, 9])
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel(ylabel)

    # Identical y-axis limits across the three panels.
    values = scale * df[metric]
    margin = 0.6 if metric == "accuracy" else 0.006
    axes[0].set_ylim(
        max(0, values.min() - margin),
        min(scale, values.max() + margin),
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=3,
        bbox_to_anchor=(0.5, 0.04), frameon=False,
    )

    fig.suptitle("KGW–KGW: held-out test performance", y=0.99)
    fig.text(
        0.5, 0.01,
        "Solid lines: whole-text + window features. "
        "Dashed lines: corresponding whole-text-only baseline.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.17, 1, 0.94))

    for extension in ("png", "pdf"):
        destination = out / f"kgw_kgw_{metric}.{extension}"
        fig.savefig(destination, dpi=300, bbox_inches="tight")
        print("Saved:", destination)

    plt.close(fig)

print("\nFinished. No training or generation repeated.")

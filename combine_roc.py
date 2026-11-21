from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs/comparison_all_methods/combined_roc"
OUT.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = {
    "KGW–KGW": ROOT / "runs/window_sweep_KGW",
    "PRW–PRW": ROOT / "runs/window_sweep_PRW_PRW",
    "KGW–PRW": ROOT / "runs/window_sweep_KGW_PRW",
}

MODELS = {
    "linear": ("Linear regression", "tab:blue"),
    "svm": ("SVM", "tab:orange"),
    "mlp": ("Neural network", "tab:green"),
}

def layout_key(name):
    if name == "whole":
        return (0, 0)
    for group, prefix in enumerate(
        ("dynamic", "fixed25x", "fixed59x"), start=1
    ):
        if name.startswith(prefix):
            return (group, int(name[len(prefix):]))
    return (4, name)

# Read and validate all saved test predictions before plotting.
curves = {}
layout_sets = []

for method, folder in EXPERIMENTS.items():
    if not folder.is_dir():
        raise SystemExit(f"Missing experiment directory: {folder}")

    layouts = {
        p.name for p in folder.iterdir()
        if p.is_dir() and (
            p.name == "whole"
            or re.fullmatch(r"dynamic\d+|fixed(?:25|59)x\d+", p.name)
        )
    }
    layout_sets.append(layouts)

    for layout in sorted(layouts, key=layout_key):
        reference = None

        for model in MODELS:
            path = folder / layout / f"test_predictions_{model}.csv"
            if not path.exists():
                raise SystemExit(f"Missing predictions: {path}")

            df = pd.read_csv(path, dtype={"idx": str})
            required = {"idx", "true_label", "score"}
            if not required.issubset(df.columns):
                raise SystemExit(
                    f"{path}: expected columns {sorted(required)}; "
                    f"found {list(df.columns)}"
                )

            y = pd.to_numeric(df["true_label"], errors="raise").to_numpy()
            scores = pd.to_numeric(df["score"], errors="raise").to_numpy()

            if set(np.unique(y)) != {0, 1}:
                raise SystemExit(f"{path}: expected labels 0 and 1")
            if not np.isfinite(scores).all():
                raise SystemExit(f"{path}: nonfinite prediction scores")
            if df.duplicated(["idx", "true_label"]).any():
                raise SystemExit(f"{path}: duplicate test examples")

            # Confirm classifiers were evaluated on the same examples.
            identity = df[["idx", "true_label"]].sort_values(
                ["idx", "true_label"]
            ).reset_index(drop=True)
            if reference is None:
                reference = identity
            elif not identity.equals(reference):
                raise SystemExit(
                    f"{method}/{layout}: classifier test examples differ"
                )

            fpr, tpr, _ = roc_curve(y, scores, pos_label=1)
            auc = roc_auc_score(y, scores)
            curves[(method, layout, model)] = (fpr, tpr, auc, len(y))

if not all(s == layout_sets[0] for s in layout_sets):
    raise SystemExit(
        "The experiments have different window layouts. "
        "Complete the missing layouts before combining."
    )

layouts = sorted(layout_sets[0], key=layout_key)
if not layouts:
    raise SystemExit("No saved window layouts found.")

def draw_panel(ax, method, layout):
    for model, (name, color) in MODELS.items():
        fpr, tpr, auc, n = curves[(method, layout, model)]
        ax.plot(
            fpr, tpr, color=color, linewidth=1.7,
            label=f"{name}: AUC={auc:.4f}"
        )

    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.set_title(f"{method} | {layout} | n={n}", fontsize=10)
    ax.legend(loc="lower right", fontsize=7)

def save(fig, name):
    for extension in ("png", "pdf"):
        fig.savefig(
            OUT / f"{name}.{extension}",
            dpi=300, bbox_inches="tight"
        )
    plt.close(fig)

# One figure per layout: method pairs are shown side by side.
for layout in layouts:
    fig, axes = plt.subplots(
        1, len(EXPERIMENTS), figsize=(15, 5), squeeze=False
    )
    for ax, method in zip(axes[0], EXPERIMENTS):
        draw_panel(ax, method, layout)

    fig.suptitle(
        f"Held-out test ROC — {layout} — positive class: B_first"
    )
    fig.supxlabel("False positive rate: A_first predicted as B_first")
    fig.supylabel("True positive rate: B_first recall")
    fig.tight_layout(rect=(0.02, 0.04, 1, 0.94))
    save(fig, f"roc_{layout}_all_methods")

# Overview figures: each row is a layout; each column a method pair.
families = {
    "whole": [x for x in layouts if x == "whole"],
    "dynamic": [x for x in layouts if x.startswith("dynamic")],
    "fixed25": [x for x in layouts if x.startswith("fixed25x")],
    "fixed59": [x for x in layouts if x.startswith("fixed59x")],
}

for family, members in families.items():
    if not members:
        continue

    fig, axes = plt.subplots(
        len(members), len(EXPERIMENTS),
        figsize=(15, 4.5 * len(members)),
        squeeze=False
    )
    for i, layout in enumerate(members):
        for j, method in enumerate(EXPERIMENTS):
            draw_panel(axes[i, j], method, layout)

    fig.suptitle(
        f"Held-out test ROC — {family} — positive class: B_first",
        y=0.995
    )
    fig.supxlabel("False positive rate: A_first predicted as B_first")
    fig.supylabel("True positive rate: B_first recall")
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.98))
    save(fig, f"roc_overview_{family}")

# Save the AUC values used in the figures.
summary = [
    {
        "method_pair": method,
        "layout": layout,
        "model": MODELS[model][0],
        "test_n": values[3],
        "auc_B_first": values[2],
    }
    for (method, layout, model), values in curves.items()
]
pd.DataFrame(summary).to_csv(OUT / "roc_auc_summary.csv", index=False)

print("Saved ROC figures and AUC summary:")
print(OUT)

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
OUT = RUNS / "comparison_all_methods"

EXPERIMENTS = {
    "KGW-KGW": "window_sweep_KGW",
    "PRW-PRW": "window_sweep_PRW_PRW",
    "KGW-PRW": "window_sweep_KGW_PRW",
}

MODEL_NAMES = {
    "linear": "LR",
    "svm": "SVM",
    "mlp": "MLP",
}

METRICS = [
    ("macro_precision", 4, 1),
    ("macro_recall", 4, 1),
    ("macro_f1", 4, 1),
    ("accuracy", 2, 100),
    ("auc", 4, 1),
]

def layout_key(value):
    value = str(value)
    if value == "whole":
        return (0, 0)
    for group, prefix in enumerate(
        ("dynamic", "fixed25x", "fixed59x"), start=1
    ):
        if value.startswith(prefix):
            return (group, int(value[len(prefix):]))
    return (4, value)

def escape(value):
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(c, c) for c in str(value))

def export_table(df, method, split, destination):
    df = df.copy().reset_index(drop=True)

    required = {
        "layout", "model",
        *(column for column, _, _ in METRICS),
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{method}/{split}: missing columns {sorted(missing)}"
        )

    # Keep only fixed59x4 among fixed 59-token layouts.
    keep = (
        ~df["layout"].astype(str).str.startswith("fixed59x")
        | df["layout"].eq("fixed59x4")
    )
    df = df.loc[keep].copy()

    # Remove these columns from exported CSVs as well.
    df = df.drop(columns=["windows", "n_features"], errors="ignore")

    if df.empty:
        raise ValueError(f"{method}/{split}: no retained rows")

    if df.duplicated(["layout", "model"]).any():
        raise ValueError(
            f"{method}/{split}: duplicate layout/model rows"
        )

    for column, _, _ in METRICS:
        df[column] = pd.to_numeric(df[column], errors="raise")
        if df[column].isna().any():
            raise ValueError(f"{method}/{split}: missing {column}")

    layouts = sorted(df["layout"].unique(), key=layout_key)
    layout_order = {layout: i for i, layout in enumerate(layouts)}
    model_order = {"linear": 0, "svm": 1, "mlp": 2}

    df["_layout_order"] = df["layout"].map(layout_order)
    df["_model_order"] = df["model"].map(model_order).fillna(99)
    df = df.sort_values(
        ["_layout_order", "_model_order", "model"]
    ).drop(columns=["_layout_order", "_model_order"])
    df = df.reset_index(drop=True)

    destination.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination / f"{split}_comparison.csv", index=False)

    method_tex = method.replace("-", "--")
    label = method.lower().replace("-", "_")

    lines = [
        r"% Requires \usepackage{graphicx}",
        r"\begin{table}[htbp]",
        r"\centering",
        (
            rf"\caption{{{split.capitalize()} Performance Comparison "
            rf"of Classification Models Across Window Layouts "
            rf"({method_tex}). Prec., Rec., and F1 are macro averages. "
            r"Bold indicates the best displayed value for each metric "
            r"within a layout; ties are bolded.}"
        ),
        rf"\label{{tab:{label}_{split}_layout_performance}}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{llccccc}",
        r"\hline",
        r"Layout & Model & Prec. & Rec. & F1 & Acc. (\%) & AUC \\",
        r"\hline",
    ]

    for layout in layouts:
        group = df[df["layout"].eq(layout)]

        displayed = {}
        best = {}
        for column, decimals, multiplier in METRICS:
            values = (group[column] * multiplier).map(
                lambda x: f"{x:.{decimals}f}"
            )
            displayed[column] = values
            best[column] = max(float(x) for x in values)

        for idx, row in group.iterrows():
            cells = [
                escape(layout),
                escape(MODEL_NAMES.get(row["model"], row["model"])),
            ]

            for column, _, _ in METRICS:
                value = displayed[column].loc[idx]
                if float(value) == best[column]:
                    value = rf"\textbf{{{value}}}"
                cells.append(value)

            lines.append(" & ".join(cells) + r" \\")

        lines.append(r"\hline")

    lines.extend([
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ])

    path = destination / f"{split}_comparison.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {path} ({len(df)} rows)")
    return df

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    for split in ("validation", "test"):
        combined = []

        for method, folder in EXPERIMENTS.items():
            source = RUNS / folder / f"{split}_comparison.csv"
            df = pd.read_csv(source)

            if "split" in df.columns:
                df = df[
                    df["split"].astype(str).str.strip().str.lower().eq(split)
                ].copy()

            if df.empty:
                raise ValueError(f"No {split} results in {source}")

            destination = OUT / method.replace("-", "_")
            df = export_table(df, method, split, destination)
            df.insert(0, "method_pair", method)
            combined.append(df)

        combined = pd.concat(combined, ignore_index=True)
        path = OUT / f"{split}_comparison_all.csv"
        combined.to_csv(path, index=False)
        print("Saved:", path)

    print("\nDone. Updated exported tables; source results remain unchanged.")

if __name__ == "__main__":
    main()

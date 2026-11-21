from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"

# Separate destination: previous exported tables remain unchanged.
OUT = RUNS / "comparison_all_methods_grouped_average_windows_only4"

EXPERIMENTS = {
    "KGW-KGW": "windows_only_three_pairs_1buCwy/KGW_KGW",
    "PRW-PRW": "windows_only_three_pairs_1buCwy/PRW_PRW",
    "KGW-PRW": "windows_only_three_pairs_1buCwy/KGW_PRW",
}

MODEL_NAMES = {
    "linear": "LR",
    "svm": "SVM",
    "mlp": "MLP",
}
MODEL_ORDER = {"linear": 0, "svm": 1, "mlp": 2}

METRICS = [
    ("macro_precision", 4, 1),
    ("macro_recall", 4, 1),
    ("macro_f1", 4, 1),
    ("accuracy", 2, 100),
    ("auc", 4, 1),
]

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

# def parse_layout(layout):
#     layout = str(layout)

#     if layout == "whole":
#         return "Whole", 0, (0, 0)

#     match = re.fullmatch(r"dynamic(\d+)", layout)
#     if match:
#         count = int(match.group(1))
#         if 4 <= count <= 9:
#             return "Dynamic", count, (1, 0)
#         return None

#     match = re.fullmatch(r"fixed(\d+)x(\d+)", layout)
#     if match:
#         width, count = map(int, match.groups())

#         if width == 25:
#             if 4 <= count <= 8:
#                 return "Fixed_25", count, (2, 25)
#             return None

#         # Keep only four windows for average-width/59-token layouts.
#         if count == 4:
#             return f"Fixed_{width}", count, (3, width)

#     return None

def parse_layout(layout):
    layout = str(layout)

    # Windows-only experiment: exclude standalone whole-text results.
    if layout == "whole":
        return None

    match = re.fullmatch(r"dynamic(\d+)", layout)
    if match:
        count = int(match.group(1))
        if 4 <= count <= 9:
            return "Dynamic", count, (1, 0)
        return None

    match = re.fullmatch(r"fixed(\d+)x(\d+)", layout)
    if match:
        width, count = map(int, match.groups())

        if width == 25:
            if 4 <= count <= 8:
                return "Fixed_25", count, (2, 25)
            return None

        # Include average-token layouts with 1, 2, 3, and 4 windows.
        if 1 <= count <= 4:
            return f"Fixed_{width}", count, (3, width)

    return None


def prepare_rows(source, method, split):
    df = pd.read_csv(source)

    required = {
        "layout", "model",
        *(column for column, _, _ in METRICS),
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{source}: missing columns {sorted(missing)}")

    if "split" in df.columns:
        df = df[
            df["split"].astype(str).str.strip().str.lower().eq(split)
        ].copy()

    if df.empty:
        raise ValueError(f"{source}: no {split} rows")

    if df.duplicated(["layout", "model"]).any():
        raise ValueError(f"{source}: duplicate layout/model rows")

    parsed = df["layout"].map(parse_layout)
    keep = parsed.notna()
    df = df.loc[keep].copy()
    parsed = parsed.loc[keep]

    if df.empty:
        raise ValueError(f"{source}: no retained layouts")

    df["_family"] = [value[0] for value in parsed]
    df["_windows"] = [value[1] for value in parsed]
    df["_family_group"] = [value[2][0] for value in parsed]
    df["_family_width"] = [value[2][1] for value in parsed]

    unknown_models = set(df["model"]) - set(MODEL_NAMES)
    if unknown_models:
        raise ValueError(f"{source}: unknown models {unknown_models}")

    for layout, group in df.groupby("layout"):
        if set(group["model"]) != set(MODEL_NAMES):
            raise ValueError(f"{source}: incomplete classifiers for {layout}")

    for column, _, _ in METRICS:
        df[column] = pd.to_numeric(df[column], errors="raise")
        if not np.isfinite(df[column].to_numpy()).all():
            raise ValueError(f"{source}: invalid values in {column}")

    df["_model_order"] = df["model"].map(MODEL_ORDER)
    df = df.sort_values([
        "_family_group", "_family_width", "_windows", "_model_order"
    ]).reset_index(drop=True)

    return df

def export(df, method, split, destination):
    destination.mkdir(parents=True, exist_ok=True)

    # Save only the columns displayed in the table.
    table = pd.DataFrame({
        "layout": df["_family"],
        "windows": df["_windows"],
        "model": df["model"].map(MODEL_NAMES),
        **{column: df[column] for column, _, _ in METRICS},
    })
    table.to_csv(destination / f"{split}_comparison.csv", index=False)

    method_tex = method.replace("-", "--")
    label = method.lower().replace("-", "_")

    lines = [
        r"% Requires \usepackage{graphicx,multirow,booktabs}",
        r"\begin{table}[htbp]",
        r"\centering",
        (
            rf"\caption{{{split.capitalize()} Performance Comparison "
            rf"of Classification Models Across Window Layouts "
            rf"({method_tex}). Fixed-layout subscripts indicate tokens "
            r"per window. Prec., Rec., and F1 are macro averages. "
            r"Bold marks the best displayed value within each "
            r"layout--window configuration; ties are bolded.}"
        ),
        rf"\label{{tab:{label}_{split}_grouped_performance}}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{cccccccc}",
        r"\hline",
        r"Layout & Windows & Model & Prec. & Rec. & F1 & Acc. (\%) & AUC \\",
        r"\hline",
    ]

    # Family spans and window spans are calculated automatically.
    for family, family_rows in df.groupby("_family", sort=False):
        family_first = True

        for window_index, (count, group) in enumerate(
            family_rows.groupby("_windows", sort=False)
        ):
            if window_index > 0:
                lines.append(r"\cmidrule(r){2-8}")

            displayed = {}
            best = {}

            for column, decimals, multiplier in METRICS:
                values = (group[column] * multiplier).map(
                    lambda x: f"{x:.{decimals}f}"
                )
                displayed[column] = values
                best[column] = max(float(value) for value in values)

            for row_index, (idx, row) in enumerate(group.iterrows()):
                family_cell = ""
                if family_first:
                    family_cell = (
                        rf"\multirow{{{len(family_rows)}}}{{*}}"
                        rf"{{{escape(family)}}}"
                    )
                    family_first = False

                windows_cell = ""
                if row_index == 0:
                    windows_cell = (
                        rf"\multirow{{{len(group)}}}{{*}}{{{int(count)}}}"
                    )

                cells = [
                    family_cell,
                    windows_cell,
                    MODEL_NAMES[row["model"]],
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
    print(f"Saved: {path} ({len(table)} rows)")
    return table

def main():
    # Validate every input before writing tables.
    prepared = {}
    for split in ("validation", "test"):
        for method, folder in EXPERIMENTS.items():
            source = RUNS / folder / f"{split}_comparison.csv"
            if not source.exists():
                raise FileNotFoundError(
                    f"Missing: {source}\n"
                    "Finish the average-width sweep or update EXPERIMENTS."
                )
            prepared[(method, split)] = prepare_rows(source, method, split)

    OUT.mkdir(parents=True, exist_ok=True)

    for split in ("validation", "test"):
        combined = []

        for method in EXPERIMENTS:
            destination = OUT / method.replace("-", "_")
            table = export(
                prepared[(method, split)],
                method, split, destination
            )
            table.insert(0, "method_pair", method)
            combined.append(table)

        path = OUT / f"{split}_comparison_all.csv"
        pd.concat(combined, ignore_index=True).to_csv(path, index=False)
        print("Saved:", path)

    print("\nDone. Source results and previous exported tables are unchanged.")

if __name__ == "__main__":
    main()

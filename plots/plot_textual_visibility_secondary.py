#!/usr/bin/env python3
"""Create the Study A textual-visibility 100% stacked-bar figure."""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "plots/.matplotlib_cache")
os.environ.setdefault("XDG_CACHE_HOME", "plots/.cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    SCRIPT_DIR.parents[1]
    / "output"
    / "kaito"
    / "large_data_creator_profile"
    / "search_engine"
    / "study_a"
    / "plots"
    / "textual_visibility_12_balanced.xlsx"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "text_visibility"

SHEET_NAME = "Raw Figure Data"
GENERATOR_ORDER = ["Mistral-Nemo", "Qwen3-8B", "GPT-OSS-20B"]
SELECTION_ORDER = ["1st", "2nd", "3rd", "4th"]
GROUP_ORDER = ["Low", "Mid", "High"]
VALUE_COLUMNS = ["Generated Share", "Textual Visibility"]
REQUIRED_COLUMNS = ["Generator", "Selection", "Method", "Prior Attention Group", *VALUE_COLUMNS]
METHOD_DISPLAY = {
    "RankNet": "RankNet",
    "EAR": "EAR",
    "EAR-Sym": "EAR-Sym",
    "No Reranking": "No Reranking",
    "Pairwise Reg": "Pairwise Reg",
    "Boratto-reg": "Boratto-reg",
    "PAL": "PAL",
    "PBiLoss": "PBiLoss",
}
GENERATOR_ALIASES = {
    "mistral": "Mistral-Nemo",
    "mistral-nemo": "Mistral-Nemo",
    "qwen": "Qwen3-8B",
    "qwen3-8b": "Qwen3-8B",
    "gpt": "GPT-OSS-20B",
    "gpt-oss": "GPT-OSS-20B",
    "gpt-oss-20b": "GPT-OSS-20B",
}

LOWER_COLOR = "#2F7FB8"
HIGH_COLOR = "#C94C4C"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.weight": "normal",
            "axes.labelweight": "normal",
            "axes.titleweight": "normal",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 300,
        }
    )


def normalize_generator(value: object) -> str:
    text = str(value).strip()
    key = text.casefold().replace("_", "-")
    return GENERATOR_ALIASES.get(key, text)


def read_raw_data(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=SHEET_NAME)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    out = df.copy()
    out["Generator"] = out["Generator"].map(normalize_generator)
    out["Selection"] = out["Selection"].astype(str).str.strip()
    out["Method"] = out["Method"].astype(str).str.strip()
    out["Prior Attention Group"] = out["Prior Attention Group"].astype(str).str.strip()
    for col in VALUE_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out[out["Generator"].isin(GENERATOR_ORDER)]
    out = out[out["Selection"].isin(SELECTION_ORDER)]
    out = out[out["Prior Attention Group"].isin(GROUP_ORDER)]
    out = out.dropna(subset=VALUE_COLUMNS)

    expected = {
        (generator, selection, group)
        for generator in GENERATOR_ORDER
        for selection in SELECTION_ORDER
        for group in GROUP_ORDER
    }
    observed = set(
        map(
            tuple,
            out[["Generator", "Selection", "Prior Attention Group"]].drop_duplicates().to_numpy(),
        )
    )
    missing_rows = sorted(expected - observed)
    if missing_rows:
        raise ValueError(f"Missing generator/selection/group rows: {missing_rows}")

    duplicated = out[out.duplicated(["Generator", "Selection", "Prior Attention Group"], keep=False)]
    if not duplicated.empty:
        detail = duplicated[["Generator", "Selection", "Prior Attention Group"]].to_dict("records")
        raise ValueError(f"Duplicate generator/selection/group rows found: {detail}")

    out["Generator"] = pd.Categorical(out["Generator"], GENERATOR_ORDER, ordered=True)
    out["Selection"] = pd.Categorical(out["Selection"], SELECTION_ORDER, ordered=True)
    out["Prior Attention Group"] = pd.Categorical(out["Prior Attention Group"], GROUP_ORDER, ordered=True)
    return out.sort_values(["Generator", "Selection", "Prior Attention Group"]).reset_index(drop=True)


def validate_share_sums(df: pd.DataFrame, tolerance: float = 0.005) -> pd.DataFrame:
    records = []
    for (generator, selection), group in df.groupby(["Generator", "Selection"], observed=True):
        record = {"Generator": str(generator), "Selection": str(selection)}
        for col in VALUE_COLUMNS:
            total = float(group[col].sum())
            record[col] = total
            if abs(total - 1.0) > tolerance:
                raise ValueError(
                    f"{col} sums to {total:.6f} for {generator}, {selection}; "
                    f"expected approximately 1.0 within {tolerance}."
                )
        records.append(record)
    return pd.DataFrame(records)


def wrap_label(label: str) -> str:
    if label == "No Reranking":
        return "No\nReranking"
    return label


def display_method(value: object) -> str:
    text = str(value).strip()
    return METHOD_DISPLAY.get(text, text)


def method_labels(panel: pd.DataFrame) -> list[str]:
    base_labels = []
    for selection in SELECTION_ORDER:
        methods = (
            panel[panel["Selection"].astype(str).eq(selection)]["Method"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        if len(methods) != 1:
            raise ValueError(
                f"Expected one Method value for {panel['Generator'].iloc[0]}, {selection}; "
                f"found {methods}."
            )
        base_labels.append(display_method(methods[0]))

    counts = Counter(base_labels)
    labels = []
    for selection, label in zip(SELECTION_ORDER, base_labels):
        display = f"{label} ({selection})" if counts[label] > 1 else label
        labels.append(wrap_label(display))
    return labels


def aggregate_for_plot(panel: pd.DataFrame, selection: str, value_col: str) -> tuple[float, float]:
    rows = panel[panel["Selection"].astype(str).eq(selection)]
    lower = rows[rows["Prior Attention Group"].astype(str).isin(["Low", "Mid"])][value_col].sum() * 100.0
    high = rows[rows["Prior Attention Group"].astype(str).eq("High")][value_col].sum() * 100.0
    total = lower + high
    if not np.isclose(total, 100.0, atol=0.5):
        raise ValueError(
            f"{value_col} lower+high totals {total:.3f}% for "
            f"{panel['Generator'].iloc[0]}, {selection}; expected 100%."
        )
    return float(lower), float(high)


def style_axis(ax: plt.Axes, show_y: bool) -> None:
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_facecolor("white")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#b8b8b8")
        ax.spines[spine].set_linewidth(0.9)

    ax.tick_params(axis="x", labelsize=10.7, width=0.8, length=3.0, pad=1)
    ax.tick_params(axis="y", labelsize=8.5, width=0.8, length=3.0)
    for label in ax.get_xticklabels():
        label.set_fontweight("normal")
    for label in ax.get_yticklabels():
        label.set_fontweight("normal")
    if not show_y:
        ax.tick_params(axis="y", left=False, labelleft=False)


def plot_stacked(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    configure_style()
    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(10, 5),
        sharey=True,
    )
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.23, top=0.80, wspace=0.14)

    x = np.arange(len(SELECTION_ORDER)) * 0.74
    bar_width = 0.12
    offsets = [-0.18, 0.18]
    bar_positions = [x_i + offset for x_i in x for offset in offsets]
    bar_labels = ["Gen.", "TV."] * len(SELECTION_ORDER)

    for col_idx, generator in enumerate(GENERATOR_ORDER):
        ax = axes[col_idx]
        panel = df[df["Generator"].astype(str).eq(generator)]
        labels = method_labels(panel)

        for sel_idx, selection in enumerate(SELECTION_ORDER):
            for value_idx, value_col in enumerate(VALUE_COLUMNS):
                lower, high = aggregate_for_plot(panel, selection, value_col)
                xpos = x[sel_idx] + offsets[value_idx]
                ax.bar(
                    xpos,
                    lower,
                    width=bar_width,
                    color=LOWER_COLOR,
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=2,
                )
                ax.bar(
                    xpos,
                    high,
                    width=bar_width,
                    bottom=lower,
                    color=HIGH_COLOR,
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=2,
                )
        ax.set_title(generator, fontsize=11.0, fontweight="normal", pad=5)
        ax.set_xlim(x[0] - 0.28, x[-1] + 0.28)
        ax.set_xticks(bar_positions)
        ax.set_xticklabels(bar_labels)
        style_axis(ax, show_y=col_idx == 0)

        for sel_idx, label in enumerate(labels):
            ax.text(
                x[sel_idx],
                -0.12,
                label,
                ha="center",
                va="top",
                fontsize=11,
                fontweight="normal",
                transform=ax.get_xaxis_transform(),
                clip_on=False,
            )

    fig.text(
        0.023,
        0.50,
        "Share (%)",
        rotation=90,
        ha="center",
        va="center",
        fontsize=9.0,
        fontweight="normal",
    )

    handles = [
        Patch(facecolor=LOWER_COLOR, edgecolor="white", label="Low Attention + Mid Attention"),
        Patch(facecolor=HIGH_COLOR, edgecolor="white", label="High Attention"),
    ]
    legend = fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.91),
        ncol=2,
        frameon=False,
        fontsize=8.8,
        handlelength=1.2,
        columnspacing=1.5,
    )
    for text in legend.get_texts():
        text.set_fontweight("normal")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "textual_visibility_stacked.pdf",
        output_dir / "textual_visibility_stacked.png",
    ]
    for path in paths:
        fig.savefig(path, bbox_inches="tight", dpi=350, facecolor="white")
    plt.close(fig)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the Study A textual-visibility stacked-bar figure.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input Excel workbook.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = read_raw_data(args.input)
    validation = validate_share_sums(df)
    paths = plot_stacked(df, args.output_dir)
    max_error = (validation[VALUE_COLUMNS] - 1.0).abs().to_numpy().max()

    print(f"Validated Low+Mid+High sums for plotted shares; max absolute error = {max_error:.6f}")
    print("Generated files:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()

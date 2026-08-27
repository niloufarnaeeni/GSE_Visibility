#!/usr/bin/env python3
"""Plot Study A creator/exposure share distributions from the 12-row Excel file."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "plots/.matplotlib_cache")
os.environ.setdefault("XDG_CACHE_HOME", "plots/.cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
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
    / "final_12_prior_attention_group_results_revised.xlsx"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output" / "share"

GENERATOR_ORDER = ["Mistral-Nemo", "Qwen3-8B", "GPT-OSS-20B"]
PRIOR_ATTENTION_GROUPS = ["Low", "Mid", "High"]
SHARE_TYPES = ["Creator Share", "Exposure Share"]

SHARE_COLUMNS = {
    "Creator Share": ["Low Creator Share", "Mid Creator Share", "High Creator Share"],
    "Exposure Share": ["Low Exposure Share", "Mid Exposure Share", "High Exposure Share"],
}

COLORS = {
    "Low": "#2F7FB8",
    "Mid": "#B9B3A8",
    "High": "#C94C4C",
}

REQUIRED_COLUMNS = [
    "Generator",
    "Order",
    "Method",
    *SHARE_COLUMNS["Creator Share"],
    *SHARE_COLUMNS["Exposure Share"],
]


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 300,
        }
    )


def shorten_loss_type(value: object) -> str:
    text = str(value).strip()
    replacements = {
        "ranknet": "RankNet",
        "ear": "EAR",
        "ear_sym": "EAR-Sym",
        "pairwise_reg": "Pairwise Reg",
        "boratto_reg": "Boratto-reg",
        "pal": "PAL",
        "pbiloss_popneg_ft": "PBiLoss",
    }
    return replacements.get(text, text.replace("pairwise_", "").replace("_", "-"))


def canonical_generator(value: object) -> str | None:
    text = str(value).strip()
    key = text.lower()
    if key.startswith("mistral"):
        return "Mistral-Nemo"
    if key.startswith("gpt"):
        return "GPT-OSS-20B"
    if key.startswith("qwen"):
        return "Qwen3-8B"
    return None


def read_share_data(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Final 12 Rows")
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    out = df.copy()
    out["Generator"] = out["Generator"].map(canonical_generator)
    out["Order"] = pd.to_numeric(out["Order"], errors="coerce").astype("Int64")
    for col in [col for cols in SHARE_COLUMNS.values() for col in cols]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out[out["Generator"].isin(GENERATOR_ORDER)]
    out = out[out["Order"].isin([1, 2, 3, 4])]
    out = out.dropna(subset=[col for cols in SHARE_COLUMNS.values() for col in cols])

    expected = {(generator, order) for generator in GENERATOR_ORDER for order in [1, 2, 3, 4]}
    observed = set(map(tuple, out[["Generator", "Order"]].drop_duplicates().to_numpy()))
    missing_pairs = sorted(expected - observed)
    if missing_pairs:
        raise ValueError(f"Missing generator/order rows: {missing_pairs}")

    duplicated = out[out.duplicated(["Generator", "Order"], keep=False)]
    if not duplicated.empty:
        detail = duplicated[["Generator", "Order"]].drop_duplicates().to_dict("records")
        raise ValueError(f"Duplicate generator/order rows found: {detail}")

    out["Generator"] = pd.Categorical(out["Generator"], GENERATOR_ORDER, ordered=True)
    out = out.sort_values(["Order", "Generator"]).reset_index(drop=True)
    return out


def normalized_percentages(row: pd.Series, columns: list[str]) -> np.ndarray:
    values = row[columns].to_numpy(dtype=float)
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"Share columns do not have a positive total for row {row.to_dict()}")
    return values / total * 100.0


def row_label_for_order(df: pd.DataFrame, order: int) -> str:
    rows = df[df["Order"].eq(order)]
    labels = []
    for _, row in rows.iterrows():
        labels.append(str(row["Method"]).strip())
    unique_methods = list(dict.fromkeys(labels))
    if len(unique_methods) == 1:
        method = unique_methods[0]
    elif len(unique_methods) == 2:
        method = f"{unique_methods[0]} / {unique_methods[1]}"
    else:
        method = f"{unique_methods[0]} / {unique_methods[1]}\n{unique_methods[2]}"

    return method.replace("No Reranking", "No\nReranking")


def add_stacked_bar(
    ax: mpl.axes.Axes,
    x0: float,
    y: float,
    width: float,
    height: float,
    percentages: np.ndarray,
) -> None:
    left = x0
    for group, pct in zip(PRIOR_ATTENTION_GROUPS, percentages):
        segment_width = width * pct / 100.0
        ax.barh(
            y,
            segment_width,
            left=left,
            height=height,
            color=COLORS[group],
            edgecolor="white",
            linewidth=0.65,
        )
        if group == "Low" and pct < 22.0:
            ax.text(
                left + max(segment_width, 0.012) + 0.012,
                y + height / 2.0 + 0.075,
                f"{pct:.1f}%",
                ha="left",
                va="center",
                fontsize=6.2,
                fontweight="normal",
                color=COLORS["Low"],
            )
        elif pct >= 22.0:
            text_color = "white" if group == "High" else "#202020"
            ax.text(
                left + segment_width / 2.0,
                y,
                f"{pct:.1f}%",
                ha="center",
                va="center",
                fontsize=6.8,
                fontweight="normal",
                color=text_color,
            )
        left += segment_width


def plot_share_grid(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    configure_style()
    fig, axes = plt.subplots(
        nrows=4,
        ncols=3,
        figsize=(10.2, 4.35),
        sharex=False,
        sharey=False,
    )
    fig.subplots_adjust(left=0.12, right=0.99, top=0.84, bottom=0.13, wspace=0.10, hspace=0.28)

    bar_width = 1.0
    gap = 0.08
    x_positions = {"Creator Share": 0.0, "Exposure Share": bar_width + gap}
    y_bar = 0.40
    bar_height = 0.30

    for col_idx, generator in enumerate(GENERATOR_ORDER):
        axes[0, col_idx].set_title(generator, fontsize=11.0, fontweight="normal", pad=10)

    for row_idx, order in enumerate([1, 2, 3, 4]):
        row_label = row_label_for_order(df, order)
        axes[row_idx, 0].text(
            -0.08,
            0.50,
            row_label,
            ha="right",
            va="center",
            multialignment="center",
            transform=axes[row_idx, 0].transAxes,
            fontsize=11.6,
            fontweight="normal",
            linespacing=1.15,
        )

        for col_idx, generator in enumerate(GENERATOR_ORDER):
            ax = axes[row_idx, col_idx]
            row = df[df["Order"].eq(order) & df["Generator"].astype(str).eq(generator)].iloc[0]

            ax.set_xlim(0, 2 * bar_width + gap)
            ax.set_ylim(0, 1)
            ax.axis("off")

            for share_type in SHARE_TYPES:
                x0 = x_positions[share_type]
                ax.text(
                    x0 + bar_width / 2,
                    0.72,
                    "Creator Share" if share_type == "Creator Share" else "Exposure Share",
                    ha="center",
                    va="center",
                    fontsize=7.4,
                    fontweight="normal",
                    color="#303030",
                )
                percentages = normalized_percentages(row, SHARE_COLUMNS[share_type])
                add_stacked_bar(ax, x0, y_bar, bar_width, bar_height, percentages)

    legend_handles = [
        Line2D([0], [0], color=COLORS[group], lw=7.5, solid_capstyle="butt", label=f"{group} Attention")
        for group in PRIOR_ATTENTION_GROUPS
    ]
    legend = fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.56, 0.965),
        ncol=3,
        frameon=False,
        fontsize=9.0,
    )
    for text in legend.get_texts():
        text.set_fontweight("normal")

    fig.canvas.draw()
    left = axes[0, 0].get_position().x0
    right = axes[0, -1].get_position().x1
    for row_idx in range(1, 4):
        upper = axes[row_idx - 1, 0].get_position().y0
        lower = axes[row_idx, 0].get_position().y1
        y = (upper + lower) / 2.0
        fig.add_artist(
            Line2D(
                [left, right],
                [y, y],
                transform=fig.transFigure,
                color="#d6d6d6",
                linewidth=0.75,
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "study_a_creator_exposure_share_grid.pdf",
        output_dir / "study_a_creator_exposure_share_grid.png",
    ]
    for path in paths:
        fig.savefig(path, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Study A creator/exposure share grid.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Final 12-row Excel file.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = read_share_data(args.input)
    paths = plot_share_grid(df, args.output_dir)
    print("Generated files:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()

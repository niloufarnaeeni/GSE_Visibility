#!/usr/bin/env python3
"""Create a compact Study A OR forest plot for a single-column paper."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "plots/.matplotlib_cache")
os.environ.setdefault("XDG_CACHE_HOME", "plots/.cache")

Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

mpl = None
plt = None
Line2D = None
NullFormatter = None
np = None
pd = None
sns = None


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = SCRIPT_DIR / "output" / "regression"

FOREST_BACKBONE = None


# ============================================================
# DISPLAY CONFIGURATION
# ============================================================

METHOD_ORDER = [
    "EAR",
    "EAR-Sym",
    "RankNet",
    "No Reranking",
]

METHOD_DISPLAY = {
    "EAR": "EAR",
    "EAR-Sym": "EAR-Sym",
    "RankNet": "RankNet",
    "No Reranking": "No\nReranking",
}

GENERATOR_ORDER = [
    "Mistral-Nemo",
    "Qwen3-8B",
    "GPT-OSS-20B",
]

GENERATOR_DISPLAY = {
    "Mistral-Nemo": "Mistral",
    "Qwen3-8B": "Qwen",
    "GPT-OSS-20B": "GPT",
}

GENERATOR_STYLE = {
    "Mistral-Nemo": {
        "color": "#0072B2",
        "marker": "o",
    },
    "Qwen3-8B": {
        "color": "#D55E00",
        "marker": "s",
    },
    "GPT-OSS-20B": {
        "color": "#009E73",
        "marker": "D",
    },
}


FOREST_PANELS = [
    (
        "Relevance",
        "OR_relevance",
        "Relevance_CI_low",
        "Relevance_CI_high",
    ),
    (
        "Earlier Input Position",
        "OR_position",
        "Position_CI_low",
        "Position_CI_high",
    ),
    (
        "Prior Attention",
        "OR_prior_attention",
        "prior attention_CI_low",
        "prior attention_CI_high",
    ),
]


NUMERIC_COLUMNS = [
    "OR_relevance",
    "Relevance_CI_low",
    "Relevance_CI_high",
    "OR_position",
    "Position_CI_low",
    "Position_CI_high",
    "OR_prior_attention",
    "prior attention_CI_low",
    "prior attention_CI_high",
]

REQUIRED_COLUMNS = [
    "Generator",
    "reranker_model_name",
    "method_name",
    *NUMERIC_COLUMNS,
]


# ============================================================
# STYLE
# ============================================================

def configure_plot_style() -> None:

    sns.set_theme(style="whitegrid", context="paper")

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "DejaVu Sans",
            ],

            "font.size": 6.5,

            "axes.titlesize": 7.8,
            "axes.titleweight": "normal",

            "axes.labelsize": 7.0,
            "axes.labelweight": "normal",

            "xtick.labelsize": 6.1,
            "ytick.labelsize": 5.5,

            "legend.fontsize": 6.2,

            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",

            "axes.linewidth": 0.70,
            "grid.linewidth": 0.35,

            "savefig.dpi": 300,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


# ============================================================
# DATA
# ============================================================

def read_regression_table(path: Path) -> pd.DataFrame:

    suffix = path.suffix.lower()

    if suffix == ".csv":

        df = pd.read_csv(path)

    elif suffix in {".xlsx", ".xls"}:

        df = pd.read_excel(
            path,
            sheet_name="Plot Data",
        )

    else:

        raise ValueError(
            f"Unsupported file type: {path.suffix}"
        )

    missing = [
        col
        for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return df


def canonical_method(value: object) -> str | None:

    text = str(value).strip()

    key = (
        text.lower()
        .replace("_", " ")
        .replace("-", " ")
    )

    key = " ".join(key.split())

    if key == "ranknet":
        return "RankNet"
    if key == "ear":
        return "EAR"
    if key == "ear sym":
        return "EAR-Sym"
    if key == "pairwise reg":
        return "Pairwise Reg"
    if key == "boratto reg":
        return "Boratto-reg"
    if key == "pal":
        return "PAL"
    if key == "pbiloss":
        return "PBiLoss"

    if key in {
        "no reranking",
        "no ranking",
        "no reranker",
    }:
        return "No Reranking"

    return None


def canonical_generator(value: object) -> str | None:

    text = str(value).strip().lower()

    if text.startswith("mistral"):
        return "Mistral-Nemo"

    if text.startswith("qwen"):
        return "Qwen3-8B"

    if text.startswith("gpt"):
        return "GPT-OSS-20B"

    return None


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:

    out = df.copy()

    out["method"] = (
        out["method_name"]
        .map(canonical_method)
    )

    out["Generator"] = (
        out["Generator"]
        .map(canonical_generator)
    )

    for col in NUMERIC_COLUMNS:

        out[col] = pd.to_numeric(
            out[col],
            errors="coerce",
        )

    out = out[
        out["method"].isin(METHOD_ORDER)
    ]

    out = out[
        out["Generator"].isin(GENERATOR_ORDER)
    ]

    out = out[
        out["reranker_model_name"].notna()
    ]

    return out


def complete_grid(
    df: pd.DataFrame,
    required_numeric_cols: Iterable[str],
) -> pd.DataFrame:

    cols = [
        "reranker_model_name",
        "method",
        "Generator",
        *required_numeric_cols,
    ]

    complete = df.dropna(
        subset=cols
    ).copy()

    for col in required_numeric_cols:

        if (
            col.startswith("OR_")
            or col.endswith("_low")
            or col.endswith("_high")
        ):

            complete = complete[
                complete[col] > 0
            ]

    return complete


def assert_no_duplicate_combinations(
    df: pd.DataFrame,
    keys: list[str],
) -> None:

    duplicated = df[
        df.duplicated(
            keys,
            keep=False,
        )
    ]

    if not duplicated.empty:

        detail = (
            duplicated[keys]
            .drop_duplicates()
            .to_dict("records")
        )

        raise ValueError(
            "Duplicate method/generator combinations "
            f"found: {detail}"
        )


def select_forest_rows(
    df: pd.DataFrame,
    requested_backbone: str | None,
) -> pd.DataFrame:

    forest_cols = []

    for _, or_col, low_col, high_col in FOREST_PANELS:

        forest_cols.extend(
            [
                or_col,
                low_col,
                high_col,
            ]
        )

    complete = complete_grid(
        df,
        forest_cols,
    )

    if requested_backbone is not None:

        selected = complete[
            complete["reranker_model_name"]
            .astype(str)
            .eq(requested_backbone)
        ].copy()

        if selected.empty:

            raise ValueError(
                "No rows found for backbone: "
                f"{requested_backbone}"
            )

    else:

        selected = complete.copy()

    expected = {
        (method, generator)
        for method in METHOD_ORDER
        for generator in GENERATOR_ORDER
    }

    observed = set(
        map(
            tuple,
            selected[
                ["method", "Generator"]
            ]
            .drop_duplicates()
            .to_numpy(),
        )
    )

    missing = expected - observed

    if missing:

        raise ValueError(
            "Missing method/generator combinations: "
            f"{sorted(missing)}"
        )

    selected = selected[
        selected[
            ["method", "Generator"]
        ]
        .apply(tuple, axis=1)
        .isin(expected)
    ].copy()

    assert_no_duplicate_combinations(
        selected,
        [
            "method",
            "Generator",
        ],
    )

    selected["method"] = pd.Categorical(
        selected["method"],
        METHOD_ORDER,
        ordered=True,
    )

    selected["Generator"] = pd.Categorical(
        selected["Generator"],
        GENERATOR_ORDER,
        ordered=True,
    )

    selected = (
        selected
        .sort_values(
            [
                "method",
                "Generator",
            ]
        )
        .reset_index(drop=True)
    )

    return selected


# ============================================================
# SAVE
# ============================================================

def save_figure(
    fig: mpl.figure.Figure,
    output_dir: Path,
    stem: str,
) -> list[Path]:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = [
        output_dir / f"{stem}.pdf",
        output_dir / f"{stem}.svg",
        output_dir / f"{stem}.png",
    ]

    for path in paths:

        fig.savefig(
            path,
            bbox_inches="tight",
            dpi=300,
            facecolor="white",
            pad_inches=0.015,
        )

    return paths


# ============================================================
# FOREST PLOT
# ============================================================

def plot_forest(
    df: pd.DataFrame,
    output_dir: Path,
    requested_backbone: str | None = None,
) -> list[Path]:

    selected = select_forest_rows(
        df,
        requested_backbone,
    )

    selected.to_csv(
        output_dir / "study_a_figure1_forest_data.csv",
        index=False,
    )

    # ========================================================
    # FIGURE SIZE
    #
    # Shorter than the previous version.
    # ========================================================

    fig = plt.figure(
        figsize=(3.45, 3.30)
    )


    # ========================================================
    # AXES
    #
    # More horizontal room is now given to the actual plots.
    # ========================================================

    ax_rel = fig.add_axes(
        [
            0.140,
            0.555,
            0.427,
            0.305,
        ]
    )

    ax_pos = fig.add_axes(
        [
            0.578,
            0.555,
            0.412,
            0.305,
        ]
    )

    # Prior Attention gets a large plotting region,
    # but starts far enough right that labels
    # never touch its left spine.
    ax_prior_attention = fig.add_axes(
        [
            0.282,
            0.130,
            0.550,
            0.315,
        ]
    )

    axes = {
        "Relevance": ax_rel,
        "Earlier Input Position": ax_pos,
        "Prior Attention": ax_prior_attention,
    }


    # ========================================================
    # METHOD / GENERATOR Y POSITIONS
    # ========================================================

    method_step = 0.80

    method_position = {
        method:
            (
                len(METHOD_ORDER)
                - 1
                - idx
            )
            * method_step

        for idx, method
        in enumerate(METHOD_ORDER)
    }

    # Keep all generator rows visually distinct within every method group.
    generator_offsets = {
        "Mistral-Nemo": -0.22,
        "Qwen3-8B": 0.00,
        "GPT-OSS-20B": 0.22,
    }


    # ========================================================
    # COMMON X LIMITS
    # ========================================================

    all_values = []

    for (
        _,
        or_col,
        low_col,
        high_col,
    ) in FOREST_PANELS:

        all_values.extend(
            selected[
                or_col
            ]
            .dropna()
            .tolist()
        )

        all_values.extend(
            selected[
                low_col
            ]
            .dropna()
            .tolist()
        )

        all_values.extend(
            selected[
                high_col
            ]
            .dropna()
            .tolist()
        )

    all_values = np.asarray(
        all_values,
        dtype=float,
    )

    all_values = all_values[
        np.isfinite(all_values)
        & (all_values > 0)
    ]

    global_min = min(
        float(all_values.min()),
        1.0,
    )

    global_max = max(
        float(all_values.max()),
        1.0,
    )

    log_range = (
        np.log(global_max)
        - np.log(global_min)
    )

    # Slightly smaller padding gives more usable width.
    log_pad = 0.04 * log_range

    common_xlim = (
        np.exp(
            np.log(global_min)
            - log_pad
        ),
        np.exp(
            np.log(global_max)
            + log_pad
        ),
    )


    # ========================================================
    # DRAW EACH PANEL
    # ========================================================

    for (
        title,
        or_col,
        low_col,
        high_col,
    ) in FOREST_PANELS:

        ax = axes[title]

        ax.set_xscale(
            "log"
        )

        ax.set_xlim(
            common_xlim
        )


        # ----------------------------------------------------
        # OR = 1
        # ----------------------------------------------------

        ax.axvline(
            1.0,
            color="0.30",
            linestyle="--",
            linewidth=0.90,
            zorder=1,
        )


        # ----------------------------------------------------
        # PANEL TITLE
        #
        # Position title is deliberately smaller.
        # ----------------------------------------------------

        if title == "Earlier Input Position":

            title_size = 7.1

        else:

            title_size = 7.7

        ax.set_title(
            title,
            fontsize=title_size,
            fontweight="normal",
            pad=5.0,
        )


        # ----------------------------------------------------
        # GRID
        # ----------------------------------------------------

        ax.grid(
            True,
            axis="x",
            color="0.88",
            linewidth=0.35,
        )

        ax.grid(
            False,
            axis="y",
        )


        # ----------------------------------------------------
        # POINTS + CIs
        # ----------------------------------------------------

        for generator in GENERATOR_ORDER:

            rows = selected[
                selected[
                    "Generator"
                ]
                .astype(str)
                .eq(generator)
            ]

            style = GENERATOR_STYLE[
                generator
            ]

            y = (
                rows[
                    "method"
                ]
                .astype(str)
                .map(method_position)
                .to_numpy(dtype=float)
            )

            y = (
                y
                + generator_offsets[
                    generator
                ]
            )

            x = (
                rows[
                    or_col
                ]
                .to_numpy(dtype=float)
            )

            lower = (
                rows[
                    low_col
                ]
                .to_numpy(dtype=float)
            )

            upper = (
                rows[
                    high_col
                ]
                .to_numpy(dtype=float)
            )

            xerr = np.vstack(
                [
                    x - lower,
                    upper - x,
                ]
            )

            ax.errorbar(
                x,
                y,
                xerr=xerr,

                fmt=style[
                    "marker"
                ],

                color=style[
                    "color"
                ],

                ecolor=style[
                    "color"
                ],

                # Larger markers
                markersize=5.8,

                markeredgecolor="#202020",
                markeredgewidth=0.65,

                # More visible confidence intervals
                elinewidth=1.40,

                capsize=2.6,
                capthick=1.10,

                linestyle="none",

                zorder=3,
            )


        # ----------------------------------------------------
        # X TICKS
        # ----------------------------------------------------

        ticks = [
            1.0,
            1.25,
            1.5,
        ]

        visible_ticks = [
            tick
            for tick in ticks
            if (
                common_xlim[0]
                <= tick
                <= common_xlim[1]
            )
        ]

        ax.set_xticks(
            visible_ticks
        )

        ax.set_xticklabels(
            [
                f"{tick:g}"
                for tick in visible_ticks
            ],
            fontsize=6.0,
            fontweight="normal",
        )

        ax.xaxis.set_minor_formatter(
            NullFormatter()
        )

        ax.tick_params(
            axis="x",
            which="major",
            length=2.2,
            pad=1.5,
        )


        # ----------------------------------------------------
        # NO NATIVE METHOD LABELS
        # ----------------------------------------------------

        ax.set_yticks(
            []
        )

        ax.tick_params(
            axis="y",
            left=False,
        )


        # ----------------------------------------------------
        # EXTRA TOP ROOM
        #
        # This separates title from highest markers.
        # ----------------------------------------------------

        ax.set_ylim(
            -0.30,
            (
                method_step
                * (
                    len(METHOD_ORDER)
                    - 1
                )
                + 0.37
            ),
        )


        # ----------------------------------------------------
        # SPINES
        # ----------------------------------------------------

        ax.spines[
            "top"
        ].set_visible(False)

        ax.spines[
            "right"
        ].set_visible(False)

        ax.spines[
            "left"
        ].set_linewidth(0.68)

        ax.spines[
            "bottom"
        ].set_linewidth(0.68)

        ax.spines[
            "left"
        ].set_color("0.72")

        ax.spines[
            "bottom"
        ].set_color("0.72")


    # ========================================================
    # METHOD LABELS
    # ========================================================

    method_labels = {
        "EAR": "EAR",
        "EAR-Sym": "EAR-Sym",
        "RankNet": "RankNet",
        "No Reranking": "No\nReranking",
    }


    def data_y_to_fig_y(
        ax,
        y,
    ):

        display_xy = (
            ax.transData
            .transform(
                (
                    1.0,
                    y,
                )
            )
        )

        fig_xy = (
            fig.transFigure
            .inverted()
            .transform(
                display_xy
            )
        )

        return fig_xy[1]


    # --------------------------------------------------------
    # RELEVANCE LABELS
    # --------------------------------------------------------

    for method in METHOD_ORDER:

        y_fig = data_y_to_fig_y(
            ax_rel,
            method_position[
                method
            ],
        )

        fig.text(
            0.092,
            y_fig,

            method_labels[
                method
            ],

            ha="center",
            va="center",

            # Smaller method labels.
            fontsize=5.0,

            fontweight="semibold",

            linespacing=0.76,
            multialignment="center",
        )


    # --------------------------------------------------------
    # PRIOR ATTENTION LABELS
    # --------------------------------------------------------

    for method in METHOD_ORDER:

        y_fig = data_y_to_fig_y(
            ax_prior_attention,
            method_position[
                method
            ],
        )

        fig.text(
            # Keep label blocks outside the Prior Attention plotting area.
            0.237,
            y_fig,

            method_labels[
                method
            ],

            ha="center",
            va="center",

            fontsize=5.0,

            fontweight="normal",

            linespacing=0.76,
            multialignment="center",
        )


    # ========================================================
    # LEGEND
    # ========================================================

    handles = []

    for generator in GENERATOR_ORDER:

        style = GENERATOR_STYLE[
            generator
        ]

        handles.append(
            Line2D(
                [0],
                [0],

                marker=style[
                    "marker"
                ],

                color="none",

                markerfacecolor=style[
                    "color"
                ],

                markeredgecolor="#202020",
                markeredgewidth=0.55,

                markersize=4.5,

                label=GENERATOR_DISPLAY[
                    generator
                ],
            )
        )

    legend = fig.legend(
        handles=handles,

        loc="upper center",

        bbox_to_anchor=(
            0.565,
            0.985,
        ),

        ncol=3,

        frameon=False,

        handletextpad=0.28,

        columnspacing=0.68,

        borderaxespad=0.0,
    )

    for text in legend.get_texts():

        text.set_fontsize(
            6.1
        )

        text.set_fontweight(
            "normal"
        )


    # ========================================================
    # COMMON X LABEL
    # ========================================================

    fig.text(
        0.557,
        0.058,

        "OR (log scale)",

        ha="center",
        va="center",

        fontsize=7.0,

        fontweight="normal",
    )


    # ========================================================
    # SAVE
    # ========================================================

    paths = save_figure(
        fig,
        output_dir,
        "study_a_or_forest_three_panel",
    )

    plt.close(
        fig
    )

    return paths


# ============================================================
# MAIN
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the Study A OR forest plot.")
    parser.add_argument("--input", required=True, help="Regression table CSV/XLSX path.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Directory for generated plot files.")
    return parser.parse_args()


def import_plot_dependencies() -> None:
    global mpl, plt, Line2D, NullFormatter, np, pd, sns
    import matplotlib as mpl_module
    import matplotlib.pyplot as plt_module
    from matplotlib.lines import Line2D as line2d_class
    from matplotlib.ticker import NullFormatter as null_formatter_class
    import numpy as np_module
    import pandas as pd_module
    import seaborn as sns_module

    mpl = mpl_module
    plt = plt_module
    Line2D = line2d_class
    NullFormatter = null_formatter_class
    np = np_module
    pd = pd_module
    sns = sns_module


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    import_plot_dependencies()
    configure_plot_style()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw = read_regression_table(
        input_path
    )

    data = prepare_data(
        raw
    )

    generated_paths = plot_forest(
        data,
        output_dir,
        FOREST_BACKBONE,
    )

    print("Generated files:")

    for path in generated_paths:

        print(path)


if __name__ == "__main__":
    main()

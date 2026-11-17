
"""Publication-ready figures for the CSE 250A baseball Bayesian network project.

Preserves the four original figure names and the two-panel content described
in the project report. Reads completed experiment CSVs; does not refit models.

From the directory containing mlb_bn/ and results/full/:
    python -m mlb_bn.plot_results --results-dir results/full
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, LogLocator, NullFormatter
import numpy as np
import pandas as pd

LABELS = {
    "majority_class": "Majority class",
    "count_only": "Count only",
    "history_1_carry": "1 pitch, carry",
    "history_1_reset": "1 pitch, reset",
    "history_2_carry": "2 pitches, carry",
    "history_2_reset": "2 pitches, reset",
    "history_3_carry": "3 pitches, carry",
    "history_3_reset": "3 pitches, reset",
}

BLUE = "#28649A"
ORANGE = "#C55D32"
GREEN = "#26866E"
PURPLE = "#8069A3"
RESET = "#829FB3"
GRAY = "#6D7985"
DARK = "#253342"
GRID = "#E5EAF0"
SPINE = "#B6C2CD"


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10,
        "axes.labelcolor": DARK,
        "axes.edgecolor": SPINE,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.color": DARK,
        "ytick.color": DARK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def style_axis(ax: plt.Axes, grid_axis: str = "x") -> None:
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3, width=0.8)


def add_panel_title(ax: plt.Axes, letter: str, title: str) -> None:
    ax.set_title(f"{letter}   {title}", loc="left", pad=12, color=DARK)


def save(fig: plt.Figure, figures: Path, name: str) -> None:
    figures.mkdir(parents=True, exist_ok=True)

    for suffix, kwargs in (
        ("png", {"dpi": 300}),
        ("pdf", {}),
    ):
        fig.savefig(
            figures / f"{name}.{suffix}",
            bbox_inches="tight",
            pad_inches=0.10,
            **kwargs,
        )

    plt.close(fig)


def read(results: Path, name: str) -> pd.DataFrame:
    path = results / name

    if not path.is_file():
        raise FileNotFoundError(f"Missing CSV: {path.resolve()}")

    return pd.read_csv(path)


def get_selected(metrics: pd.DataFrame) -> str:
    flags = (
        metrics.selected_by_validation.astype(str).str.lower()
        == "true"
    )
    matches = metrics.loc[flags, "model"]

    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one validation-selected model"
        )

    return str(matches.iloc[0])


def model_comparison(results: Path, figures: Path) -> None:
    """Figure 2: absolute log-loss and relative improvement over count-only."""

    metrics = read(results, "test_metrics.csv")
    selected = get_selected(metrics)

    baseline = float(
        metrics.loc[
            metrics.model == "count_only", "log_loss"
        ].iloc[0]
    )

    order = ["majority_class", "count_only"]

    for length in (1, 2, 3):
        for strategy in ("carry", "reset"):
            order.append(f"history_{length}_{strategy}")

    data = metrics.set_index("model").loc[order].reset_index()

    history = data.loc[
        data.model.str.startswith("history_")
    ].copy()

    history["delta"] = 10000 * (
        history.log_loss - baseline
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, 4.7),
        gridspec_kw={"width_ratios": [1.05, 1]},
    )

    fig.subplots_adjust(
        left=0.16,
        right=0.97,
        bottom=0.21,
        top=0.82,
        wspace=0.48,
    )

    fig.suptitle(
        "Does recent pitch history help?",
        fontsize=14,
        fontweight="bold",
        color=DARK,
        y=0.96,
    )

    # Panel A: Absolute log-loss comparison
    ax = axes[0]
    positions = np.arange(len(data))

    for idx, row in data.iterrows():

        if row.model == selected:
            color = ORANGE
        elif row.model == "majority_class":
            color = GRAY
        else:
            color = BLUE

        marker = "D" if row.model == selected else "o"
        size = 78 if row.model == selected else 52

        ax.scatter(
            row.log_loss,
            idx,
            s=size,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )

    ax.axvline(
        baseline,
        color=GRAY,
        linestyle=(0, (4, 3)),
        linewidth=1.15,
        label="Count-only reference",
        zorder=2,
    )

    ax.set_yticks(
        positions,
        [LABELS[x] for x in data.model],
    )

    ax.set_ylim(len(data) - 0.5, -0.5)

    min_loss = float(data.log_loss.min())
    max_loss = float(data.log_loss.max())

    ax.set_xlim(
        min_loss - 0.00035,
        max_loss + 0.00035,
    )

    ax.xaxis.set_major_formatter(
        FormatStrFormatter("%.3f")
    )

    ax.set_xlabel(
        "2025 binary log-loss (lower is better)"
    )

    add_panel_title(
        ax,
        "A",
        "All models and baseline",
    )

    style_axis(ax)

    ax.legend(
        frameon=False,
        loc="lower right",
        fontsize=8,
    )

    # Panel B: Log-loss improvement relative to count-only
    ax = axes[1]
    y = np.arange(len(history))

    colors = []

    for model in history.model:
        if model == selected:
            colors.append(ORANGE)
        elif model.endswith("carry"):
            colors.append(BLUE)
        else:
            colors.append(RESET)

    ax.barh(
        y,
        history.delta,
        height=0.64,
        color=colors,
        zorder=3,
    )

    ax.axvline(
        0,
        color=DARK,
        linewidth=1.0,
    )

    ax.set_yticks(
        y,
        [LABELS[x] for x in history.model],
    )

    ax.set_ylim(
        len(history) - 0.55,
        -0.55,
    )

    xmin = min(
        -1.85,
        float(history.delta.min()) - 0.23,
    )

    ax.set_xlim(xmin, 0.12)

    for i, row in enumerate(history.itertuples()):

        weight = (
            "bold"
            if row.model == selected
            else "normal"
        )

        ax.text(
            row.delta / 2,
            i,
            f"{row.delta:.2f}",
            color="white",
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight=weight,
            zorder=4,
        )

    ax.set_xlabel(
        "Log-loss minus count only (× 10,000)"
    )

    add_panel_title(
        ax,
        "B",
        "History models vs. count only",
    )

    style_axis(ax)

    fig.text(
        0.57,
        0.065,
        "Negative values indicate improvement; "
        "orange = validation-selected model.",
        color=GRAY,
        fontsize=8.2,
    )

    save(fig, figures, "model_comparison")


def calibration(results: Path, figures: Path) -> None:
    """Figure 3: original 10-bin calibration, preserving sparse tail bins."""

    bins = read(results, "test_calibration.csv")
    metrics = read(results, "test_metrics.csv")

    selected = get_selected(metrics)

    fig, ax = plt.subplots(
        figsize=(6.7, 5.6)
    )

    fig.subplots_adjust(
        left=0.16,
        right=0.95,
        bottom=0.19,
        top=0.86,
    )

    fig.suptitle(
        "2025 probability calibration",
        color=DARK,
        fontsize=14,
        fontweight="bold",
        y=0.97,
    )

    # Perfect calibration reference
    ax.plot(
        [0.15, 0.55],
        [0.15, 0.55],
        color=GRAY,
        linestyle=(0, (5, 4)),
        linewidth=1.35,
        zorder=1,
    )

    # Draw majority baseline first to reduce marker occlusion.
    styles = [
        ("majority_class", GREEN, "D", 2),
        ("count_only", BLUE, "o", 3),
        (selected, ORANGE, "s", 4),
    ]

    for name, color, marker, zorder in styles:

        subset = bins.loc[
            bins.model == name
        ].sort_values("mean_predicted_ball")

        sizes = 28 + 150 * np.sqrt(
            subset.n / float(bins.n.max())
        )

        ax.scatter(
            subset.mean_predicted_ball,
            subset.observed_ball_rate,
            s=sizes,
            marker=marker,
            facecolors=color,
            edgecolors="white",
            linewidths=0.8,
            alpha=0.90,
            zorder=zorder,
        )

    sparse = bins.loc[
        (bins.model == selected) & (bins.n < 100)
    ]

    for row in sparse.itertuples():

        if int(row.n) == 48:
            offset = (8, -6)
            ha = "left"
        elif int(row.n) == 13:
            offset = (-6, -4)
            ha = "right"
        else:
            offset = (8, -8)
            ha = "left"

        ax.annotate(
            f"n={int(row.n)}",
            xy=(row.mean_predicted_ball, row.observed_ball_rate),
            xytext=offset,
            textcoords="offset points",
            ha=ha,
            va="top",
            fontsize=7.8,
            color=ORANGE,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.75,
                "boxstyle": "round,pad=0.08",
            },
        )

    ax.set_xlim(0.145, 0.55)
    ax.set_ylim(0.145, 0.55)

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_xticks(
        np.arange(0.15, 0.551, 0.05)
    )

    ax.set_yticks(
        np.arange(0.15, 0.551, 0.05)
    )

    ax.set_xlabel(
        "Mean predicted probability of Ball"
    )

    ax.set_ylabel(
        "Observed Ball frequency"
    )

    style_axis(ax, "both")

    # Consistent-size legend markers.
    legend_handles = [
        Line2D(
            [0], [0],
            color=GRAY,
            lw=1.5,
            ls="--",
            label="Perfect calibration",
        ),
        Line2D(
            [0], [0],
            color=BLUE,
            marker="o",
            linestyle="none",
            markersize=8,
            label="Count only",
        ),
        Line2D(
            [0], [0],
            color=ORANGE,
            marker="s",
            linestyle="none",
            markersize=8,
            label=LABELS[selected],
        ),
        Line2D(
            [0], [0],
            color=GREEN,
            marker="D",
            linestyle="none",
            markersize=8,
            label="Majority class",
        ),
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor=GRID,
        framealpha=0.96,
        fontsize=8.6,
        borderpad=0.7,
    )

    fig.text(
        0.16,
        0.075,
        "10 equal-width bins; marker area reflects pitches per bin.",
        fontsize=8.4,
        color=GRAY,
    )

    save(fig, figures, "calibration")


def inference(results: Path, figures: Path) -> None:
    """Figure 4: future pitches and three at-bat outcomes."""

    future = read(results, "future_inference.csv")

    exact = future.loc[
        future.method == "exact_elimination"
    ].sort_values("horizon")

    ab = read(results, "at_bat_inference.csv")

    at_bat = ab.loc[
        ab.method == "exact_absorbing_chain"
    ].iloc[0]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4.35),
        gridspec_kw={"width_ratios": [1.13, 1]},
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.80,
        bottom=0.20,
        wspace=0.27,
    )

    fig.suptitle(
        "Exact inference for the selected model",
        fontsize=14,
        fontweight="bold",
        color=DARK,
        y=0.965,
    )

    # Panel A: Future pitch inference
    ax = axes[0]

    ax.plot(
        exact.horizon,
        exact.p_ball_given_pitch_exists,
        marker="o",
        ms=6.4,
        lw=2.1,
        color=BLUE,
        label="P(Ball | pitch exists)",
    )

    ax.plot(
        exact.horizon,
        exact.p_target_pitch_exists,
        marker="s",
        ms=6.4,
        lw=2.1,
        color=ORANGE,
        label="P(pitch exists)",
    )

    ax.set_xlim(0.85, 4.15)
    ax.set_ylim(0, 1.06)
    ax.set_xticks([1, 2, 3, 4])

    ax.set_xlabel(
        "Future pitch number"
    )

    ax.set_ylabel(
        "Probability"
    )

    add_panel_title(
        ax,
        "A",
        "Future pitch from a 0–0 count",
    )

    style_axis(ax, "both")

    ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=8.4,
    )

    # Panel B: At-bat termination probabilities
    ax = axes[1]

    names = [
        "Walk",
        "Strikeout",
        "Ball in play",
    ]

    probabilities = [
        float(at_bat.p_walk),
        float(at_bat.p_strikeout),
        float(at_bat.p_in_play),
    ]

    bars = ax.bar(
        names,
        probabilities,
        color=[BLUE, ORANGE, GREEN],
        width=0.58,
        zorder=3,
    )

    ax.bar_label(
        bars,
        labels=[
            f"{value:.3f}"
            for value in probabilities
        ],
        padding=4,
        fontsize=10,
        fontweight="semibold",
        color=DARK,
    )

    ax.set_ylim(0, 0.80)

    ax.set_ylabel(
        "Probability"
    )

    add_panel_title(
        ax,
        "B",
        "Current at-bat outcome",
    )

    style_axis(ax, "y")

    fig.text(
        0.08,
        0.078,
        "Initial state: 0–0 count, empty history. "
        "Future Ball probabilities are conditional on the pitch existing.",
        fontsize=8.2,
        color=GRAY,
    )

    save(fig, figures, "inference")


def sampling(results: Path, figures: Path) -> None:
    """Figure 5: forward-sampling mean error and runtime."""

    data = read(results, "sampling_summary.csv")

    series = [
        (
            "future_ball_k1",
            "Future Ball, k=1",
            BLUE,
            "o",
        ),
        (
            "future_ball_k4",
            "Future Ball, k=4",
            ORANGE,
            "s",
        ),
        (
            "at_bat_walk",
            "Walk",
            GREEN,
            "^",
        ),
        (
            "at_bat_strikeout",
            "Strikeout",
            PURPLE,
            "D",
        ),
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.0, 4.65),
    )

    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        bottom=0.20,
        top=0.73,
        wspace=0.27,
    )

    fig.suptitle(
        "Forward sampling: accuracy and runtime",
        fontsize=14,
        fontweight="bold",
        color=DARK,
        y=0.975,
    )

    # Draw query results on both panels.
    for key, label, color, marker in series:

        subset = data.loc[
            data["query"] == key
        ].sort_values("n_samples")

        if subset.empty:
            raise ValueError(
                f"Missing query in sampling_summary.csv: {key}"
            )

        fields = [
            "mean_absolute_error",
            "mean_runtime_seconds",
        ]

        for ax, field in zip(axes, fields):

            ax.plot(
                subset.n_samples,
                subset[field],
                color=color,
                marker=marker,
                markersize=6.1,
                linewidth=1.9,
                label=label,
                zorder=3,
            )

    # Panel A: Accuracy
    axes[0].axhline(
        0.02,
        color=GRAY,
        linestyle=(0, (5, 3)),
        lw=1.15,
        zorder=2,
    )

    upper_error = max(
        0.043,
        float(data.mean_absolute_error.max()) * 1.12,
    )

    axes[0].set_ylim(
        0,
        upper_error,
    )

    axes[0].set_ylabel(
        "Mean absolute error"
    )

    add_panel_title(
        axes[0],
        "A",
        "Accuracy across 5 seeded runs",
    )

    # Panel B: Runtime
    axes[1].set_yscale("log")

    axes[1].set_ylabel(
        "Mean runtime (seconds)"
    )

    add_panel_title(
        axes[1],
        "B",
        "Computation time",
    )

    # Shared axis formatting
    for ax in axes:

        ax.set_xscale("log")
        ax.set_xlim(85, 11800)

        ax.set_xticks([
            100,
            1000,
            10000,
        ])

        ax.set_xticklabels([
            "100",
            "1,000",
            "10,000",
        ])

        ax.xaxis.set_minor_locator(
            LogLocator(base=10, subs=[])
        )

        ax.xaxis.set_minor_formatter(
            NullFormatter()
        )

        ax.set_xlabel(
            "Forward samples"
        )

        style_axis(ax, "both")

    # One shared legend for both panels.
    legend_handles = []

    for key, label, color, marker in series:

        handle = Line2D(
            [0], [0],
            color=color,
            marker=marker,
            lw=1.9,
            ms=6,
            label=label,
        )

        legend_handles.append(handle)

    legend_handles.append(
        Line2D(
            [0], [0],
            color=GRAY,
            lw=1.15,
            ls="--",
            label="Mean MAE = 0.02",
        )
    )

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.865),
        ncol=5,
        frameon=False,
        fontsize=8.0,
        handlelength=2.0,
        columnspacing=1.2,
    )

    fig.text(
        0.09,
        0.067,
        "Mean of 5 seeded runs; the 0.02 threshold "
        "applies to mean error, not every run.",
        color=GRAY,
        fontsize=8.2,
    )

    save(fig, figures, "sampling")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/full"),
    )

    args = parser.parse_args()

    set_style()

    figures = args.results_dir / "figures"

    for plot in (
        model_comparison,
        calibration,
        inference,
        sampling,
    ):
        plot(args.results_dir, figures)

    print(
        f"Saved 4 PNG and 4 PDF figures to "
        f"{figures.resolve()}"
    )


if __name__ == "__main__":
    main()

"""Create report-ready figures from a completed experiment directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
BLUE = "#2171b5"
ORANGE = "#d95f02"
GRAY = "#777777"


def save(fig: plt.Figure, directory: Path, stem: str) -> None:
    fig.savefig(directory / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def model_comparison(results: Path, figures: Path) -> None:
    metrics = pd.read_csv(results / "test_metrics.csv")
    metrics = metrics.sort_values("log_loss", ascending=False)
    count_loss = float(metrics.loc[metrics.model == "count_only", "log_loss"].iloc[0])
    selected = str(metrics.loc[metrics.selected_by_validation, "model"].iloc[0])
    colors = [ORANGE if name == selected else GRAY if name == "majority_class" else BLUE
              for name in metrics.model]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), gridspec_kw={"width_ratios": [1.1, 1]})
    y = np.arange(len(metrics))
    axes[0].scatter(metrics.log_loss, y, color=colors, s=70, zorder=3)
    axes[0].set_yticks(y, [LABELS[name] for name in metrics.model])
    axes[0].invert_yaxis()
    axes[0].set_xlim(0.643, 0.654)
    axes[0].set_xlabel("2025 binary log-loss (lower is better)")
    axes[0].set_title("All models and baseline")
    axes[0].axvline(count_loss, color="black", linewidth=0.8, linestyle="--")

    history = metrics[metrics.model.str.startswith("history_")].copy()
    history["delta"] = (history.log_loss - count_loss) * 10_000
    history = history.sort_values("delta", ascending=False)
    y2 = np.arange(len(history))
    axes[1].barh(
        y2, history.delta,
        color=[ORANGE if name == selected else BLUE for name in history.model],
        height=0.72,
    )
    axes[1].set_yticks(y2, [LABELS[name] for name in history.model])
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Log-loss minus count-only (× 10,000)")
    axes[1].set_title("History models vs. count only")
    axes[1].text(0.02, -0.14, "Negative values improve on count only.", transform=axes[1].transAxes, fontsize=9)
    fig.suptitle("Does recent pitch history help?", fontsize=15, fontweight="bold")
    fig.tight_layout()
    save(fig, figures, "model_comparison")


def calibration(results: Path, figures: Path) -> None:
    bins = pd.read_csv(results / "test_calibration.csv")
    metrics = pd.read_csv(results / "test_metrics.csv")
    selected = str(metrics.loc[metrics.selected_by_validation, "model"].iloc[0])
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.plot([0, 1], [0, 1], color=GRAY, linestyle="--", label="Perfect calibration")
    for name, color, marker in [
        ("count_only", BLUE, "o"),
        (selected, ORANGE, "s"),
        ("majority_class", "#228b63", "D"),
    ]:
        subset = bins[bins.model == name].sort_values("mean_predicted_ball")
        marker_sizes = 25 + 175 * np.sqrt(subset.n / bins.n.max())
        ax.scatter(subset.mean_predicted_ball, subset.observed_ball_rate,
                   marker=marker, s=marker_sizes, color=color, label=LABELS[name], alpha=0.85)
    ax.set(xlim=(0.15, 0.55), ylim=(0.15, 0.55),
           xlabel="Predicted probability of Ball", ylabel="Observed Ball frequency",
           title="2025 probability calibration (10 bins)")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left")
    ax.text(0.02, -0.13, "Marker size reflects pitches per bin; outer bins have few pitches.",
            transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    save(fig, figures, "calibration")


def inference(results: Path, figures: Path) -> None:
    future = pd.read_csv(results / "future_inference.csv")
    exact = future[future.method == "exact_elimination"].sort_values("horizon")
    at_bat = pd.read_csv(results / "at_bat_inference.csv").iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7))
    axes[0].plot(exact.horizon, exact.p_ball_given_pitch_exists, "o-", color=BLUE,
                 label="P(Ball | target exists)")
    axes[0].plot(exact.horizon, exact.p_target_pitch_exists, "s-", color=ORANGE,
                 label="P(target pitch exists)")
    axes[0].set(xticks=[1, 2, 3, 4], ylim=(0, 1.05), xlabel="Pitch horizon",
                ylabel="Probability", title="Future pitch from a 0–0 count")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(alpha=0.2)

    events = ["Walk", "Strikeout", "Ball in play"]
    probabilities = [at_bat.p_walk, at_bat.p_strikeout, at_bat.p_in_play]
    bars = axes[1].bar(events, probabilities, color=[BLUE, ORANGE, "#228b63"])
    axes[1].bar_label(bars, fmt="%.3f", padding=3)
    axes[1].set(ylim=(0, 0.78), ylabel="Probability", title="Current at-bat outcome")
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Exact inference for the selected model", fontsize=15, fontweight="bold")
    fig.tight_layout()
    save(fig, figures, "inference")


def sampling(results: Path, figures: Path) -> None:
    data = pd.read_csv(results / "sampling_summary.csv")
    series = [
        ("future_ball_k1", "Future Ball, k=1", BLUE, "o"),
        ("future_ball_k4", "Future Ball, k=4", ORANGE, "s"),
        ("at_bat_walk", "Walk", "#228b63", "^"),
        ("at_bat_strikeout", "Strikeout", "#8e44ad", "D"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7))
    for key, label, color, marker in series:
        group = data[data["query"] == key].sort_values("n_samples")
        axes[0].plot(group.n_samples, group.mean_absolute_error, marker=marker,
                     color=color, label=label)
        axes[1].plot(group.n_samples, group.mean_runtime_seconds, marker=marker,
                     color=color, label=label)
    axes[0].axhline(0.02, color=GRAY, linestyle="--", linewidth=1, label="0.02 target")
    axes[0].set(xscale="log", xlabel="Forward samples", ylabel="Mean absolute error",
                title="Accuracy across 5 seeded runs")
    axes[1].set(xscale="log", yscale="log", xlabel="Forward samples",
                ylabel="Mean runtime (seconds)", title="Computation time")
    for ax in axes:
        ax.grid(alpha=0.2, which="both")
    axes[0].legend(fontsize=8)
    fig.suptitle("Forward sampling: accuracy and runtime", fontsize=15, fontweight="bold")
    fig.tight_layout()
    save(fig, figures, "sampling")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/full"))
    args = parser.parse_args()
    figures = args.results_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for plot in (model_comparison, calibration, inference, sampling):
        plot(args.results_dir, figures)
    print(f"Figures saved in {figures}")


if __name__ == "__main__":
    main()

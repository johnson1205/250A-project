"""Train and compare the proposal's conditional probability table models.

Run ``python -m mlb_bn.experiment --quick`` for a one-week-per-season trial, or
omit ``--quick`` for all available regular-season pitches in 2023--2025.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import OUTCOMES
from .data import load_season, prepare_pitches
from .inference import (
    State,
    at_bat_exact,
    at_bat_sample,
    future_ball_enumeration,
    future_ball_exact,
    future_ball_sample,
)
from .model import CPTModel


def model_configs() -> list[tuple[int, str]]:
    """The count-only model and both history definitions at lengths one to three."""
    return [(0, "carry")] + [
        (length, strategy)
        for length in (1, 2, 3)
        for strategy in ("carry", "reset")
    ]


def model_name(length: int, strategy: str) -> str:
    return "count_only" if length == 0 else f"history_{length}_{strategy}"


def binary_metrics(
    actual_ball: np.ndarray, p_ball: np.ndarray, hard_ball: bool | None = None
) -> dict[str, float]:
    """Evaluate Ball as the positive class, with a 0.5 hard-label threshold."""
    y = np.asarray(actual_ball, dtype=bool)
    p = np.asarray(p_ball, dtype=float)
    if len(y) == 0 or len(y) != len(p):
        raise ValueError("Metrics need equally sized, nonempty labels and probabilities")
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("Predicted probabilities must be finite and in [0, 1]")
    safe = np.clip(p, 1e-15, 1 - 1e-15)
    predicted = p >= 0.5 if hard_ball is None else np.full(len(y), hard_ball)
    tp = int(np.sum(predicted & y))
    fp = int(np.sum(predicted & ~y))
    fn = int(np.sum(~predicted & y))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    log_loss = -np.mean(y * np.log(safe) + (~y) * np.log1p(-safe))
    return {
        "n": int(len(y)),
        "log_loss": float(log_loss),
        "f1_ball": float(f1),
        "precision_ball": float(precision),
        "recall_ball": float(recall),
        "accuracy": float(np.mean(predicted == y)),
        "brier_score": float(np.mean((p - y) ** 2)),
        "ece_10": expected_calibration_error(y, p),
    }


def calibration_bins(actual_ball: np.ndarray, p_ball: np.ndarray) -> pd.DataFrame:
    """Ten equal-width bins; empty bins are omitted from the output."""
    y = np.asarray(actual_ball, dtype=bool)
    p = np.asarray(p_ball, dtype=float)
    indices = np.minimum((p * 10).astype(int), 9)
    rows = []
    for bin_index in range(10):
        mask = indices == bin_index
        if mask.any():
            rows.append(
                {
                    "bin": bin_index,
                    "lower": bin_index / 10,
                    "upper": (bin_index + 1) / 10,
                    "n": int(mask.sum()),
                    "mean_predicted_ball": float(p[mask].mean()),
                    "observed_ball_rate": float(y[mask].mean()),
                }
            )
    return pd.DataFrame(rows)


def expected_calibration_error(actual_ball: np.ndarray, p_ball: np.ndarray) -> float:
    bins = calibration_bins(actual_ball, p_ball)
    if bins.empty:
        return float("nan")
    gaps = abs(bins["mean_predicted_ball"] - bins["observed_ball_rate"])
    return float((gaps * bins["n"]).sum() / bins["n"].sum())


def four_way_log_loss(actual: pd.Series, probabilities: np.ndarray) -> float:
    indices = pd.Categorical(actual, categories=OUTCOMES).codes
    if np.any(indices < 0):
        raise ValueError("An outcome is outside the four modeled categories")
    selected = probabilities[np.arange(len(indices)), indices]
    return float(-np.log(np.clip(selected, 1e-15, 1)).mean())


def probabilities_for(model: CPTModel, frame: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(model.predict_frame(frame), dtype=float)
    if probabilities.shape != (len(frame), 4):
        raise ValueError("The CPT must produce one four-outcome distribution per pitch")
    if not np.all(np.isfinite(probabilities)) or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-10
    ):
        raise ValueError("Every CPT distribution must be finite and sum to one")
    return probabilities


def season_dates(year: int, quick: bool) -> tuple[str, str]:
    if quick:
        return f"{year}-04-01", f"{year}-04-07"
    return f"{year}-03-01", f"{year}-11-15"


def load_datasets(cache_dir: Path, quick: bool, chunk_days: int) -> tuple[dict[int, pd.DataFrame], dict]:
    seasons = {}
    reports = {}
    for year in (2023, 2024, 2025):
        start, end = season_dates(year, quick)
        raw = load_season(
            year, cache_dir, start_date=start, end_date=end, chunk_days=chunk_days
        )
        prepared, report = prepare_pitches(raw)
        if prepared.empty:
            raise RuntimeError(
                f"No modeled regular-season pitches for {year} ({start} through {end}). "
                "Check the downloaded data and date range."
            )
        seasons[year] = prepared
        reports[str(year)] = {
            "date_start": start,
            "date_end": end,
            "raw_rows": int(len(raw)),
            "modeled_rows": int(len(prepared)),
            "exclusions": report,
        }
        print(f"{year}: {len(prepared):,} modeled pitches", flush=True)
    return seasons, reports


def choose_models(
    train: pd.DataFrame, validation: pd.DataFrame, alphas: list[float]
) -> tuple[pd.DataFrame, dict[str, tuple[CPTModel, float]]]:
    """Select each model's alpha using only 2024 binary log-loss."""
    actual = (validation["outcome"] == "Ball").to_numpy()
    rows = []
    chosen = {}
    for length, strategy in model_configs():
        name = model_name(length, strategy)
        best_loss = float("inf")
        best_model = None
        best_alpha = None
        for alpha in alphas:
            model = CPTModel(history_length=length, strategy=strategy, alpha=alpha)
            model.fit(train)
            probabilities = probabilities_for(model, validation)
            metrics = binary_metrics(actual, probabilities[:, 0])
            metrics["four_way_log_loss"] = four_way_log_loss(
                validation["outcome"], probabilities
            )
            rows.append(
                {
                    "model": name,
                    "history_length": length,
                    "history_strategy": "none" if length == 0 else strategy,
                    "alpha": alpha,
                    **metrics,
                }
            )
            if metrics["log_loss"] < best_loss:
                best_loss = metrics["log_loss"]
                best_model = model
                best_alpha = alpha
        assert best_model is not None and best_alpha is not None
        chosen[name] = best_model, best_alpha
        print(f"{name}: validation log-loss {best_loss:.5f} at alpha={best_alpha:g}", flush=True)
    return pd.DataFrame(rows), chosen


def evaluate_test(
    train: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    chosen: dict[str, tuple[CPTModel, float]],
    test: pd.DataFrame,
    output_dir: Path,
) -> tuple[CPTModel, str]:
    """Evaluate all selected CPTs on 2025; save pitch predictions for the winner."""
    selected_row = validation_metrics.sort_values(
        ["log_loss", "model", "alpha"], kind="stable"
    ).iloc[0]
    selected_name = str(selected_row["model"])
    actual = (test["outcome"] == "Ball").to_numpy()
    rows = []
    calibration_rows = []

    training_ball_rate = float((train["outcome"] == "Ball").mean())
    baseline_probability = np.full(len(test), training_ball_rate)
    baseline_hard_ball = training_ball_rate > 0.5  # Tie resolves to NotBall.
    baseline_metrics = binary_metrics(actual, baseline_probability, hard_ball=baseline_hard_ball)
    rows.append(
        {
            "model": "majority_class",
            "history_length": 0,
            "history_strategy": "none",
            "alpha": np.nan,
            "selected_by_validation": False,
            "hard_prediction": "Ball" if baseline_hard_ball else "NotBall",
            "training_ball_rate": training_ball_rate,
            "four_way_log_loss": np.nan,
            **baseline_metrics,
        }
    )
    baseline_bins = calibration_bins(actual, baseline_probability)
    baseline_bins.insert(0, "model", "majority_class")
    calibration_rows.append(baseline_bins)

    selected_model = chosen[selected_name][0]
    for length, strategy in model_configs():
        name = model_name(length, strategy)
        model, alpha = chosen[name]
        probabilities = probabilities_for(model, test)
        metrics = binary_metrics(actual, probabilities[:, 0])
        rows.append(
            {
                "model": name,
                "history_length": length,
                "history_strategy": "none" if length == 0 else strategy,
                "alpha": alpha,
                "selected_by_validation": name == selected_name,
                "hard_prediction": "Ball if p_ball >= 0.5",
                "training_ball_rate": np.nan,
                "four_way_log_loss": four_way_log_loss(test["outcome"], probabilities),
                **metrics,
            }
        )
        bins = calibration_bins(actual, probabilities[:, 0])
        bins.insert(0, "model", name)
        calibration_rows.append(bins)
        if name == selected_name:
            id_columns = [
                name
                for name in (
                    "game_pk", "game_date", "pitcher", "at_bat_number",
                    "pitch_number", "balls", "strikes", "description", "outcome",
                )
                if name in test.columns
            ]
            predictions = test[id_columns].copy().reset_index(drop=True)
            predictions.insert(0, "model", selected_name)
            predictions["actual_ball"] = actual.astype(int)
            for index, outcome in enumerate(OUTCOMES):
                predictions[f"p_{outcome.lower()}"] = probabilities[:, index]
            predictions["p_not_ball"] = 1 - probabilities[:, 0]
            predictions["predicted_binary"] = np.where(
                probabilities[:, 0] >= 0.5, "Ball", "NotBall"
            )
            predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    pd.DataFrame(rows).to_csv(output_dir / "test_metrics.csv", index=False)
    pd.concat(calibration_rows, ignore_index=True).to_csv(
        output_dir / "test_calibration.csv", index=False
    )
    return selected_model, selected_name


def save_interpretation(
    chosen: dict[str, tuple[CPTModel, float]], output_dir: Path
) -> None:
    """Compare the proposal's two ordered paths to a 2-1 count."""
    model, alpha = chosen["history_3_carry"]
    rows = []
    for chronological in (("Ball", "Ball", "Strike"), ("Strike", "Ball", "Ball")):
        history = tuple(reversed(chronological))
        probabilities = model.predict_proba(2, 1, history)
        rows.append(
            {
                "balls": 2,
                "strikes": 1,
                "chronological_history": "-".join(chronological),
                "history_most_recent_first": "-".join(history),
                "alpha": alpha,
                **{f"p_{outcome.lower()}": probabilities[index]
                   for index, outcome in enumerate(OUTCOMES)},
                "p_not_ball": 1 - probabilities[0],
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "history_interpretation.csv", index=False)


def run_inference_experiment(
    model: CPTModel,
    output_dir: Path,
    seed: int,
    sample_sizes: tuple[int, ...] = (100, 1_000, 10_000),
    repetitions: int = 5,
) -> None:
    """Compare exact inference, enumeration, and repeated forward sampling.

    The query is a fresh 0-0 at-bat with no prior pitcher history.  For each
    sample size, five fixed-seed repetitions show how sampling error changes.
    Future-pitch probabilities are conditional on the target pitch existing.
    """
    state = State(0, 0, (None,) * model.history_length)
    if not sample_sizes or any(size < 1 for size in sample_sizes) or repetitions < 1:
        raise ValueError("sample_sizes and repetitions must be positive")
    future_rows = []
    at_bat_rows = []

    for horizon in range(1, 5):
        started = time.perf_counter()
        exact = future_ball_exact(model, state, horizon)
        exact_seconds = time.perf_counter() - started
        started = time.perf_counter()
        enumerated = future_ball_enumeration(model, state, horizon)
        enumeration_seconds = time.perf_counter() - started
        if not np.isclose(exact.existence_probability, enumerated.existence_probability, atol=1e-12):
            raise AssertionError("Exact state propagation and enumeration disagree on survival")
        if exact.ball_probability is not None and not np.isclose(
            exact.ball_probability, enumerated.ball_probability, atol=1e-12
        ):
            raise AssertionError("Exact state propagation and enumeration disagree on Ball probability")

        for method, result, seconds in (
            ("exact_elimination", exact, exact_seconds),
            ("brute_force_enumeration", enumerated, enumeration_seconds),
        ):
            future_rows.append(
                {
                    "horizon": horizon,
                    "method": method,
                    "n_samples": 0,
                    "repetition": 0,
                    "p_ball_given_pitch_exists": result.ball_probability,
                    "p_target_pitch_exists": result.existence_probability,
                    "absolute_ball_error": 0.0,
                    "absolute_existence_error": 0.0,
                    "runtime_seconds": seconds,
                }
            )

        for size in sample_sizes:
            for repetition in range(repetitions):
                started = time.perf_counter()
                sampled = future_ball_sample(
                    model, state, horizon, size,
                    seed=seed + 100_000 * horizon + 10_000 * repetition + size,
                )
                seconds = time.perf_counter() - started
                future_rows.append(
                    {
                        "horizon": horizon,
                        "method": "forward_sampling",
                        "n_samples": size,
                        "repetition": repetition + 1,
                        "p_ball_given_pitch_exists": sampled.ball_probability,
                        "p_target_pitch_exists": sampled.existence_probability,
                        "absolute_ball_error": (
                            abs(sampled.ball_probability - exact.ball_probability)
                            if sampled.ball_probability is not None and exact.ball_probability is not None
                            else np.nan
                        ),
                        "absolute_existence_error": abs(
                            sampled.existence_probability - exact.existence_probability
                        ),
                        "runtime_seconds": seconds,
                    }
                )

    started = time.perf_counter()
    exact_at_bat = at_bat_exact(model, state)
    at_bat_seconds = time.perf_counter() - started
    at_bat_rows.append(
        {
            "method": "exact_absorbing_chain",
            "n_samples": 0,
            "repetition": 0,
            "p_walk": exact_at_bat.walk,
            "p_strikeout": exact_at_bat.strikeout,
            "p_in_play": exact_at_bat.in_play,
            "p_unresolved": exact_at_bat.unresolved,
            "absolute_walk_error": 0.0,
            "absolute_strikeout_error": 0.0,
            "runtime_seconds": at_bat_seconds,
        }
    )
    for size in sample_sizes:
        for repetition in range(repetitions):
            started = time.perf_counter()
            sampled = at_bat_sample(
                model, state, size, seed=seed + 1_000_000 + 10_000 * repetition + size
            )
            seconds = time.perf_counter() - started
            at_bat_rows.append(
                {
                    "method": "forward_sampling",
                    "n_samples": size,
                    "repetition": repetition + 1,
                    "p_walk": sampled.walk,
                    "p_strikeout": sampled.strikeout,
                    "p_in_play": sampled.in_play,
                    "p_unresolved": sampled.unresolved,
                    "absolute_walk_error": abs(sampled.walk - exact_at_bat.walk),
                    "absolute_strikeout_error": abs(sampled.strikeout - exact_at_bat.strikeout),
                    "runtime_seconds": seconds,
                }
            )

    future = pd.DataFrame(future_rows)
    at_bat = pd.DataFrame(at_bat_rows)
    future.to_csv(output_dir / "future_inference.csv", index=False)
    at_bat.to_csv(output_dir / "at_bat_inference.csv", index=False)

    # The sample threshold is descriptive: the smallest tested size whose
    # mean absolute error across five repetitions is at most two points.
    summary = []
    sampled_future = future[future["method"] == "forward_sampling"]
    for (horizon, size), group in sampled_future.groupby(["horizon", "n_samples"], sort=True):
        summary.append(
            {
                "query": f"future_ball_k{horizon}",
                "horizon": horizon,
                "n_samples": size,
                "mean_absolute_error": group["absolute_ball_error"].mean(),
                "mean_runtime_seconds": group["runtime_seconds"].mean(),
                "repetitions": repetitions,
            }
        )
    sampled_at_bat = at_bat[at_bat["method"] == "forward_sampling"]
    for size, group in sampled_at_bat.groupby("n_samples", sort=True):
        for event in ("walk", "strikeout"):
            summary.append(
                {
                    "query": f"at_bat_{event}",
                    "horizon": np.nan,
                    "n_samples": size,
                    "mean_absolute_error": group[f"absolute_{event}_error"].mean(),
                    "mean_runtime_seconds": group["runtime_seconds"].mean(),
                    "repetitions": repetitions,
                }
            )
    summary_frame = pd.DataFrame(summary)
    summary_frame["mean_error_at_most_0.02"] = summary_frame["mean_absolute_error"] <= 0.02
    summary_frame.to_csv(output_dir / "sampling_summary.csv", index=False)
    requirements = []
    for query, group in summary_frame.groupby("query", sort=False):
        acceptable = group.loc[group["mean_error_at_most_0.02"], "n_samples"]
        requirements.append(
            {
                "query": query,
                "smallest_tested_n_with_mean_error_at_most_0.02": (
                    int(acceptable.min()) if len(acceptable) else np.nan
                ),
            }
        )
    pd.DataFrame(requirements).to_csv(output_dir / "sampling_requirement.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--quick", action="store_true",
        help="Use April 1--7 of each season instead of the complete seasons",
    )
    parser.add_argument("--chunk-days", type=int, default=7)
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=[0.1, 1.0, 10.0],
        help="Laplace smoothing values selected on 2024 validation log-loss",
    )
    parser.add_argument("--seed", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_days < 1:
        raise ValueError("--chunk-days must be positive")
    if not args.alphas or any(not np.isfinite(a) or a < 0 for a in args.alphas):
        raise ValueError("--alphas must contain nonnegative finite values")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    seasons, reports = load_datasets(args.cache_dir, args.quick, args.chunk_days)
    train, validation, test = (seasons[year] for year in (2023, 2024, 2025))
    validation_metrics, chosen = choose_models(train, validation, args.alphas)
    validation_metrics.to_csv(args.output_dir / "validation_metrics.csv", index=False)
    selected_model, selected_name = evaluate_test(
        train, validation_metrics, chosen, test, args.output_dir
    )
    save_interpretation(chosen, args.output_dir)
    run_inference_experiment(selected_model, args.output_dir, args.seed)
    (args.output_dir / "data_report.json").write_text(
        json.dumps(reports, indent=2, default=str), encoding="utf-8"
    )
    run_info = {
        "train_year": 2023,
        "validation_year": 2024,
        "test_year": 2025,
        "quick": args.quick,
        "alpha_grid": args.alphas,
        "selected_model_by_2024_log_loss": selected_name,
        "selected_alpha": chosen[selected_name][1],
        "seed": args.seed,
        "elapsed_seconds_before_inference": time.perf_counter() - start_time,
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_info, indent=2), encoding="utf-8"
    )
    print(f"Selected {selected_name}; results saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

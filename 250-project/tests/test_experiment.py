"""One small integration check for selection, evaluation, and inference files."""

import numpy as np
import pandas as pd

from mlb_bn.data import prepare_pitches
from mlb_bn.experiment import (
    choose_models,
    evaluate_test,
    run_inference_experiment,
    save_interpretation,
)


def _season(year: int) -> pd.DataFrame:
    observations = [
        (1, 1, 10, 0, 0, "ball"),
        (1, 2, 10, 1, 0, "called_strike"),
        (1, 3, 10, 1, 1, "foul"),
        (1, 4, 10, 1, 2, "hit_into_play"),
        (2, 1, 10, 0, 0, "called_strike"),
        (2, 2, 10, 0, 1, "ball"),
        (2, 3, 10, 1, 1, "swinging_strike"),
        (2, 4, 10, 1, 2, "foul"),
        (2, 5, 10, 1, 2, "ball"),
        (2, 6, 10, 2, 2, "ball"),
        (2, 7, 10, 3, 2, "ball"),
        (3, 1, 11, 0, 0, "hit_into_play"),
    ]
    raw = pd.DataFrame(
        [
            {
                "game_pk": year * 10,
                "game_date": f"{year}-04-01",
                "pitcher": pitcher,
                "at_bat_number": at_bat,
                "pitch_number": pitch,
                "balls": balls,
                "strikes": strikes,
                "description": description,
                "game_type": "R",
            }
            for at_bat, pitch, pitcher, balls, strikes, description in observations
        ]
    )
    return prepare_pitches(raw)[0]


def test_complete_experiment_on_tiny_local_fixture(tmp_path):
    train, validation, test = (_season(year) for year in (2023, 2024, 2025))
    validation_metrics, chosen = choose_models(train, validation, [0.1, 1.0])
    assert len(validation_metrics) == 14
    assert len(chosen) == 7

    selected_model, selected_name = evaluate_test(
        train, validation_metrics, chosen, test, tmp_path
    )
    assert selected_name in chosen
    predictions = pd.read_csv(tmp_path / "test_predictions.csv")
    assert len(predictions) == len(test)
    assert np.allclose(predictions["p_ball"] + predictions["p_not_ball"], 1)
    assert len(pd.read_csv(tmp_path / "test_metrics.csv")) == 8

    save_interpretation(chosen, tmp_path)
    assert len(pd.read_csv(tmp_path / "history_interpretation.csv")) == 2
    run_inference_experiment(
        selected_model, tmp_path, seed=250, sample_sizes=(10, 30), repetitions=2
    )
    future = pd.read_csv(tmp_path / "future_inference.csv")
    assert set(future["horizon"]) == {1, 2, 3, 4}
    assert set(future["method"]) == {
        "exact_elimination", "brute_force_enumeration", "forward_sampling"
    }
    at_bat = pd.read_csv(tmp_path / "at_bat_inference.csv")
    assert np.isclose(at_bat.iloc[0][["p_walk", "p_strikeout", "p_in_play"]].sum(), 1)
    assert (tmp_path / "sampling_requirement.csv").exists()

"""Focused checks for Statcast chronology, classification, and chunk caching."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pandas as pd

from mlb_bn.data import load_season, prepare_pitches


def _pitch(
    game: int,
    at_bat: int,
    number: int,
    pitcher: int,
    description: str,
    balls: int = 0,
    strikes: int = 0,
    game_type: str = "R",
) -> dict:
    return {
        "game_pk": game,
        "game_date": "2023-04-01" if game == 1 else "2023-04-02",
        "pitcher": pitcher,
        "at_bat_number": at_bat,
        "pitch_number": number,
        "balls": balls,
        "strikes": strikes,
        "description": description,
        "game_type": game_type,
    }


def test_history_carries_across_at_bats_but_not_exclusions_or_pitchers():
    rows = [
        _pitch(1, 1, 1, 10, "ball"),
        _pitch(1, 1, 2, 10, "foul", 1, 0),
        _pitch(1, 1, 3, 10, "hit_by_pitch", 1, 1),
        _pitch(1, 2, 1, 10, "called_strike"),
        _pitch(1, 2, 2, 10, "ball", 0, 1),
        _pitch(1, 2, 3, 10, "foul_tip", 1, 1),
        _pitch(1, 3, 1, 10, "foul_bunt"),
        _pitch(1, 4, 1, 11, "ball"),
        _pitch(1, 5, 1, 10, "ball"),
        _pitch(2, 1, 1, 10, "ball"),
        _pitch(2, 2, 1, 10, "ball", game_type="P"),
    ]
    # A source fetch need not arrive in chronological order.
    prepared, report = prepare_pitches(pd.DataFrame(rows[::-1]))

    assert len(prepared) == 9
    assert report["input_rows"] == 11
    assert report["non_regular_season"] == 1
    assert report["unsupported_description"] == 1
    assert report["excluded_by_description"] == {"hit_by_pitch": 1}
    assert report["excluded_total"] == 2

    a1p2 = prepared.query("game_pk == 1 and at_bat_number == 1 and pitch_number == 2").iloc[0]
    assert a1p2["h1_carry"] == "Ball"
    assert a1p2["h1_reset"] == "Ball"

    a2p1 = prepared.query("game_pk == 1 and at_bat_number == 2 and pitch_number == 1").iloc[0]
    assert a2p1["h1_carry"] is None
    assert a2p1["h1_reset"] is None

    a3p1 = prepared.query("game_pk == 1 and at_bat_number == 3 and pitch_number == 1").iloc[0]
    assert (a3p1["h1_carry"], a3p1["h2_carry"], a3p1["h3_carry"]) == (
        "Strike", "Ball", "Strike"
    )
    assert (a3p1["h1_reset"], a3p1["h2_reset"], a3p1["h3_reset"]) == (
        None, None, None
    )
    assert a3p1["outcome"] == "Strike"  # foul bunts can be strike three

    for game, at_bat in [(1, 4), (1, 5), (2, 1)]:
        row = prepared.query("game_pk == @game and at_bat_number == @at_bat").iloc[0]
        assert row["h1_carry"] is None


def test_automatic_call_and_invalid_count_interrupt_history():
    rows = [
        _pitch(1, 1, 1, 10, "ball"),
        _pitch(1, 1, 2, 10, "automatic_strike", 1, 0),
        _pitch(1, 1, 3, 10, "called_strike", 1, 1),
        _pitch(1, 1, 4, 10, "foul", 4, 2),
        _pitch(1, 1, 5, 10, "ball", 1, 2),
    ]
    prepared, report = prepare_pitches(pd.DataFrame(rows))
    assert report["unsupported_description"] == 1
    assert report["invalid_count"] == 1
    assert report["excluded_total"] == 2
    assert prepared.iloc[1]["h1_carry"] is None
    assert prepared.iloc[2]["h1_carry"] is None


def test_missing_pitch_number_interrupts_history():
    rows = [
        _pitch(1, 1, 1, 10, "ball"),
        _pitch(1, 1, 3, 10, "called_strike", 1, 1),
        _pitch(1, 2, 2, 10, "ball", 0, 1),
    ]
    prepared, _ = prepare_pitches(pd.DataFrame(rows))
    assert prepared.iloc[1]["h1_carry"] is None
    assert prepared.iloc[2]["h1_carry"] is None


def test_season_chunks_are_reused_for_a_subset(tmp_path, monkeypatch):
    calls: list[tuple[str, str]] = []

    def fake_statcast(*, start_dt, end_dt, verbose, parallel):
        calls.append((start_dt, end_dt))
        day = "2023-01-02" if start_dt == "2023-01-01" else "2023-01-08"
        return pd.DataFrame([
            {**_pitch(1, 1, 1, 10, "ball"), "game_date": pd.Timestamp(day)},
            {**_pitch(2, 1, 1, 10, "ball", game_type="P"), "game_date": pd.Timestamp(day)},
        ])

    monkeypatch.setitem(sys.modules, "pybaseball", SimpleNamespace(statcast=fake_statcast))
    first = load_season(2023, tmp_path, "2023-01-01", "2023-01-08")
    assert calls == [("2023-01-01", "2023-01-07"), ("2023-01-08", "2023-01-14")]
    assert len(first) == 2
    assert set(first["game_type"]) == {"R"}

    subset = load_season(2023, tmp_path, "2023-01-02", "2023-01-03")
    assert len(calls) == 2  # No second download for the overlapping chunk.
    assert len(subset) == 1
    assert str(subset.iloc[0]["game_date"]).startswith("2023-01-02")

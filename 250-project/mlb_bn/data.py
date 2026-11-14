"""Download Statcast pitches and construct leakage-free pitcher histories.

Only four pitch outcomes are modeled.  Rows with another description (including
hit-by-pitch and pitch-clock automatic calls) are excluded, and both history
buffers are cleared at their position in the original pitch sequence.  This
prevents two modeled pitches from becoming artificial neighbors.
"""

from __future__ import annotations

from collections import deque
from datetime import date, timedelta
from pathlib import Path
from uuid import uuid4

import pandas as pd

from .constants import OUTCOMES


REQUIRED_COLUMNS = (
    "game_pk",
    "game_date",
    "pitcher",
    "at_bat_number",
    "pitch_number",
    "balls",
    "strikes",
    "description",
    "game_type",
)
OPTIONAL_COLUMNS = ("events", "inning", "inning_topbot", "type")

# The categories reflect how a pitch changes the count in the simplified
# at-bat model.  A foul tip and a foul bunt are strikes even with two strikes;
# an ordinary foul is not a third strike.  A foul tip on a bunt is also a
# strike.  In-play variants in older Statcast exports are kept for compatibility.
DESCRIPTION_TO_OUTCOME = {
    "ball": "Ball",
    "blocked_ball": "Ball",
    "intent_ball": "Ball",
    "pitchout": "Ball",
    "called_strike": "Strike",
    "swinging_strike": "Strike",
    "swinging_strike_blocked": "Strike",
    "missed_bunt": "Strike",
    "swinging_pitchout": "Strike",
    "foul_tip": "Strike",
    "foul_bunt": "Strike",
    "bunt_foul_tip": "Strike",
    "foul": "Foul",
    "foul_pitchout": "Foul",
    "hit_into_play": "InPlay",
    "hit_into_play_no_out": "InPlay",
    "hit_into_play_score": "InPlay",
    "pitchout_hit_into_play": "InPlay",
    "pitchout_hit_into_play_no_out": "InPlay",
    "pitchout_hit_into_play_score": "InPlay",
}

if set(DESCRIPTION_TO_OUTCOME.values()) != set(OUTCOMES):
    raise RuntimeError("Description mapping and modeled outcomes disagree")


def _date_in_year(value: str | None, year: int, default: date) -> date:
    result = date.fromisoformat(value) if value is not None else default
    if result.year != year:
        raise ValueError(f"Date {result} is outside season year {year}")
    return result


def load_season(
    year: int,
    cache_dir: Path,
    start_date: str | None = None,
    end_date: str | None = None,
    chunk_days: int = 7,
) -> pd.DataFrame:
    """Load a calendar year's regular-season Statcast data, caching chunks.

    ``start_date`` and ``end_date`` allow a small test run.  Chunk boundaries
    are anchored to January 1, so a subset and a later full-season run reuse
    the same cached files.  Both date endpoints are inclusive.  Downloads are
    cached only after successful retrieval; failures propagate to the caller.

    The cache keeps the fields needed by this project, rather than all of the
    much wider Statcast export.  The final DataFrame contains regular-season
    rows only; exclusion of unsupported pitch descriptions is done separately
    by :func:`prepare_pitches` so those exclusions can be counted and can break
    the pitch history at the correct position.
    """
    if chunk_days < 1:
        raise ValueError("chunk_days must be at least 1")
    first_day = date(year, 1, 1)
    last_day = date(year, 12, 31)
    requested_start = _date_in_year(start_date, year, first_day)
    requested_end = _date_in_year(end_date, year, last_day)
    if requested_start > requested_end:
        raise ValueError("start_date must be on or before end_date")

    season_dir = Path(cache_dir) / str(year)
    season_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    first_chunk = (requested_start - first_day).days // chunk_days
    last_chunk = (requested_end - first_day).days // chunk_days

    for chunk_index in range(first_chunk, last_chunk + 1):
        chunk_start = first_day + timedelta(days=chunk_index * chunk_days)
        chunk_end = min(chunk_start + timedelta(days=chunk_days - 1), last_day)
        cache_path = season_dir / f"statcast_{chunk_start}_{chunk_end}.csv.gz"
        if cache_path.exists():
            chunk = pd.read_csv(cache_path, low_memory=False)
        else:
            # Import lazily so cached analysis and unit tests do not require
            # pybaseball or a working network connection.
            try:
                from pybaseball import statcast
            except ImportError as exc:
                raise RuntimeError(
                    "pybaseball is required to download uncached Statcast data"
                ) from exc

            chunk = statcast(
                start_dt=chunk_start.isoformat(),
                end_dt=chunk_end.isoformat(),
                verbose=False,
                parallel=True,
            )
            if chunk is None:
                raise RuntimeError(f"Statcast returned no data object for {chunk_start} to {chunk_end}")
            if not isinstance(chunk, pd.DataFrame):
                raise TypeError("Statcast did not return a pandas DataFrame")
            if chunk.empty:
                chunk = pd.DataFrame(columns=REQUIRED_COLUMNS)
            else:
                missing = set(REQUIRED_COLUMNS) - set(chunk.columns)
                if missing:
                    raise ValueError(f"Statcast data lacks required columns: {sorted(missing)}")
                chunk = chunk.loc[chunk["game_type"].eq("R"), [
                    column for column in (*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS)
                    if column in chunk.columns
                ]].copy()

            temporary_path = cache_path.with_name(f"{cache_path.name}.{uuid4().hex}.tmp.gz")
            try:
                chunk.to_csv(temporary_path, index=False)
                temporary_path.replace(cache_path)
            finally:
                temporary_path.unlink(missing_ok=True)

        if not chunk.empty:
            missing = set(REQUIRED_COLUMNS) - set(chunk.columns)
            if missing:
                raise ValueError(f"Cached Statcast data lacks required columns: {sorted(missing)}")
            frames.append(chunk.loc[chunk["game_type"].eq("R")])

    if not frames:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    result = pd.concat(frames, ignore_index=True)
    game_dates = pd.to_datetime(result["game_date"], errors="coerce").dt.date
    if game_dates.isna().any():
        raise ValueError("Statcast data includes an invalid game_date")
    result = result.loc[game_dates.between(requested_start, requested_end)].copy()
    return result.reset_index(drop=True)


def prepare_pitches(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Classify pitches and add the previous three outcomes for each strategy.

    ``h1`` means the immediately preceding pitch, ``h2`` the one before that,
    and ``h3`` the third preceding pitch.  The carry strategy follows the
    current pitcher's contiguous appearance in a game across at-bats; the
    reset strategy clears at each at-bat.  Both clear when the pitcher changes,
    a new game begins, an excluded pitch/event occurs, or a pitch-number gap
    shows that a row is missing.  Unavailable history is represented by Python
    ``None``.

    Rows without the game, pitcher, at-bat, or pitch ordering fields cannot be
    placed safely in a chronology and cause an error rather than an invented
    history.  Duplicate pitch identifiers likewise cause an error.

    The report assigns each excluded row one reason: non-regular season,
    unsupported description, or invalid pre-pitch count, in that order.
    ``excluded_by_description`` gives the unsupported description frequencies
    after filtering to regular-season games; missing descriptions use
    ``<missing>``.
    """
    missing = set(REQUIRED_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Pitch data lacks required columns: {sorted(missing)}")

    report: dict = {
        "input_rows": len(raw),
        "non_regular_season": 0,
        "unsupported_description": 0,
        "invalid_count": 0,
        "excluded_by_description": {},
        "excluded_total": 0,
        "retained_rows": 0,
    }
    history_columns = [
        f"h{lag}_{strategy}"
        for strategy in ("carry", "reset")
        for lag in (1, 2, 3)
    ]
    regular = raw["game_type"].eq("R").fillna(False)
    report["non_regular_season"] = int((~regular).sum())
    work = raw.loc[regular].copy()
    if work.empty:
        empty = work.assign(outcome=pd.Series(dtype="object"))
        for column in history_columns:
            empty[column] = pd.Series(dtype="object")
        report["excluded_total"] = report["input_rows"]
        return empty.reset_index(drop=True), report

    ordering_fields = ("game_pk", "game_date", "pitcher", "at_bat_number", "pitch_number")
    if work.loc[:, list(ordering_fields)].isna().any().any():
        raise ValueError("Cannot order pitches: a game, date, pitcher, at-bat, or pitch ID is missing")
    work["game_date"] = pd.to_datetime(work["game_date"], errors="coerce")
    work["at_bat_number"] = pd.to_numeric(work["at_bat_number"], errors="coerce")
    work["pitch_number"] = pd.to_numeric(work["pitch_number"], errors="coerce")
    if (
        work["game_date"].isna().any()
        or work["at_bat_number"].isna().any()
        or work["pitch_number"].isna().any()
        or (work["at_bat_number"] < 1).any()
        or (work["pitch_number"] < 1).any()
        or (work["at_bat_number"] % 1 != 0).any()
        or (work["pitch_number"] % 1 != 0).any()
    ):
        raise ValueError("Cannot order pitches: an invalid game date, at-bat, or pitch number exists")
    work["at_bat_number"] = work["at_bat_number"].astype(int)
    work["pitch_number"] = work["pitch_number"].astype(int)
    work["_game_key"] = work["game_pk"].astype(str)
    work["_pitcher_key"] = work["pitcher"].astype(str)

    pitch_ids = ["_game_key", "at_bat_number", "pitch_number"]
    if work.duplicated(pitch_ids).any():
        raise ValueError("Duplicate game/at-bat/pitch identifiers would corrupt pitch history")

    work = work.sort_values(
        ["game_date", "_game_key", "at_bat_number", "pitch_number"], kind="stable"
    ).reset_index(drop=True)
    descriptions = work["description"].astype("string").str.strip().str.lower()
    work["outcome"] = descriptions.map(DESCRIPTION_TO_OUTCOME)
    unsupported = work["outcome"].isna()
    report["unsupported_description"] = int(unsupported.sum())
    report["excluded_by_description"] = {
        str(description): int(count)
        for description, count in descriptions.loc[unsupported]
        .fillna("<missing>")
        .value_counts()
        .items()
    }

    work["balls"] = pd.to_numeric(work["balls"], errors="coerce")
    work["strikes"] = pd.to_numeric(work["strikes"], errors="coerce")
    valid_count = (
        work["balls"].notna()
        & work["strikes"].notna()
        & work["balls"].between(0, 3)
        & work["strikes"].between(0, 2)
        & work["balls"].mod(1).eq(0)
        & work["strikes"].mod(1).eq(0)
    )
    report["invalid_count"] = int((~unsupported & ~valid_count).sum())
    valid = ~unsupported & valid_count

    # Iterate over the original order, including excluded rows.  A simple
    # deque makes each history value inspectable and avoids using future rows.
    carry: deque[str] = deque(maxlen=3)
    reset: deque[str] = deque(maxlen=3)
    previous_game: str | None = None
    previous_pitcher: str | None = None
    previous_at_bat: int | None = None
    previous_pitch_number: int | None = None
    histories: list[tuple[str | None, ...]] = []
    for game, pitcher, at_bat, pitch_number, outcome, is_valid in zip(
        work["_game_key"],
        work["_pitcher_key"],
        work["at_bat_number"],
        work["pitch_number"],
        work["outcome"],
        valid,
    ):
        if game != previous_game or pitcher != previous_pitcher:
            carry.clear()
            reset.clear()
        elif at_bat != previous_at_bat:
            reset.clear()
            if pitch_number != 1 or at_bat != previous_at_bat + 1:
                # A missing pitch or whole at-bat makes carried history unsafe.
                carry.clear()
        elif previous_pitch_number is not None and pitch_number != previous_pitch_number + 1:
            # A missing row in an at-bat cannot be classified or put in history.
            carry.clear()
            reset.clear()

        if is_valid:
            histories.append(
                tuple(buffer[-lag] if len(buffer) >= lag else None
                      for buffer in (carry, reset) for lag in (1, 2, 3))
            )
            carry.append(outcome)
            reset.append(outcome)
        else:
            histories.append((None,) * 6)
            carry.clear()
            reset.clear()

        previous_game = game
        previous_pitcher = pitcher
        previous_at_bat = at_bat
        previous_pitch_number = pitch_number

    for position, column in enumerate(history_columns):
        work[column] = [history[position] for history in histories]

    result = work.loc[valid].drop(columns=["_game_key", "_pitcher_key"]).copy()
    result["balls"] = result["balls"].astype(int)
    result["strikes"] = result["strikes"].astype(int)
    report["retained_rows"] = len(result)
    report["excluded_total"] = report["input_rows"] - report["retained_rows"]
    return result.reset_index(drop=True), report

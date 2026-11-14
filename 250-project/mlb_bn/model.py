"""A small discrete conditional probability table for Statcast pitch outcomes.

The only learned quantities are counts from the training season.  A prediction
uses the longest observed (count, history) configuration.  If that exact
configuration was not seen during training, it drops the oldest history item
until it finds an observed configuration.  An unseen count uses the training
season's overall outcome frequencies.  This backoff never consults validation
or test outcomes.
"""

from __future__ import annotations

from math import isfinite
from typing import Iterable

import numpy as np
import pandas as pd

from .constants import OUTCOMES


_OUTCOME_INDEX = {name: index for index, name in enumerate(OUTCOMES)}


def _count_value(value: object, maximum: int, name: str) -> int:
    """Convert a Statcast pre-pitch count, rejecting missing/invalid values."""
    if isinstance(value, (str, bool)) or pd.isna(value):
        raise ValueError(f"{name} must be an integer from 0 through {maximum}: {value!r}")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer from 0 through {maximum}: {value!r}") from exc
    if result != value or not 0 <= result <= maximum:
        raise ValueError(f"{name} must be an integer from 0 through {maximum}: {value!r}")
    return result


def _history_value(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    if value not in _OUTCOME_INDEX:
        raise ValueError(f"Unknown pitch history outcome: {value!r}")
    return str(value)


class CPTModel:
    """Estimate ``P(outcome | balls, strikes, recent outcomes)`` by counts.

    ``history`` is ordered from most recent to oldest.  ``None`` denotes an
    unavailable previous pitch.  ``alpha=0`` gives unsmoothed MLE for observed
    states; positive ``alpha`` applies four-category Laplace smoothing.
    """

    def __init__(
        self,
        history_length: int = 0,
        strategy: str = "carry",
        alpha: float = 1.0,
    ) -> None:
        if isinstance(history_length, bool) or not isinstance(history_length, int) or not 0 <= history_length <= 3:
            raise ValueError("history_length must be an integer from 0 through 3")
        if strategy not in {"carry", "reset"}:
            raise ValueError("strategy must be 'carry' or 'reset'")
        if not isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")
        self.history_length = history_length
        self.strategy = strategy
        self.alpha = float(alpha)
        self.counts_by_depth: list[dict[tuple, list[int]]] = []
        self.outcome_counts: tuple[int, ...] = ()
        self.n_training = 0
        self._fitted = False

    @property
    def history_columns(self) -> tuple[str, ...]:
        return tuple(f"h{i}_{self.strategy}" for i in range(1, self.history_length + 1))

    @property
    def class_frequency(self) -> dict[str, float]:
        """Training-only binary class frequencies for the probability baseline."""
        self._require_fitted()
        ball = self.outcome_counts[0] / self.n_training
        return {"Ball": ball, "NotBall": 1.0 - ball}

    @property
    def majority_class(self) -> str:
        """Training-only majority hard label; ties resolve to ``NotBall``."""
        frequencies = self.class_frequency
        return "Ball" if frequencies["Ball"] > frequencies["NotBall"] else "NotBall"

    def fit(self, frame: pd.DataFrame) -> CPTModel:
        """Count four outcomes for each count and history prefix in ``frame``."""
        columns = ("balls", "strikes", "outcome", *self.history_columns)
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Training data missing columns: {', '.join(missing)}")
        if frame.empty:
            raise ValueError("Cannot fit a CPT on an empty training set")

        counts: list[dict[tuple, list[int]]] = [dict() for _ in range(self.history_length + 1)]
        totals = [0] * len(OUTCOMES)
        for row in frame.loc[:, columns].itertuples(index=False, name=None):
            balls = _count_value(row[0], 3, "balls")
            strikes = _count_value(row[1], 2, "strikes")
            outcome = row[2]
            if outcome not in _OUTCOME_INDEX:
                raise ValueError(f"Unknown training pitch outcome: {outcome!r}")
            history = tuple(_history_value(value) for value in row[3:])
            index = _OUTCOME_INDEX[outcome]
            totals[index] += 1
            for depth, table in enumerate(counts):
                state = (balls, strikes, *history[:depth])
                table.setdefault(state, [0] * len(OUTCOMES))[index] += 1

        self.counts_by_depth = counts
        self.outcome_counts = tuple(totals)
        self.n_training = sum(totals)
        self._fitted = True
        return self

    def predict_proba(
        self,
        balls: int,
        strikes: int,
        history: Iterable[str | None] = (),
    ) -> tuple[float, float, float, float]:
        """Return probabilities in the order defined by ``OUTCOMES``.

        For an unseen full state, drop the oldest history item, then back off
        to count alone.  For an unseen count, use the training-wide frequencies.
        """
        self._require_fitted()
        balls = _count_value(balls, 3, "balls")
        strikes = _count_value(strikes, 2, "strikes")
        recent = tuple(_history_value(value) for value in history)
        if len(recent) != self.history_length:
            raise ValueError(f"Expected {self.history_length} history outcomes, got {len(recent)}")

        for depth in range(self.history_length, -1, -1):
            state = (balls, strikes, *recent[:depth])
            observed = self.counts_by_depth[depth].get(state)
            if observed is not None:
                return self._smooth(observed)
        return self._smooth(self.outcome_counts)

    def predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict four-category probabilities for all rows in original order."""
        self._require_fitted()
        columns = ("balls", "strikes", *self.history_columns)
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Prediction data missing columns: {', '.join(missing)}")

        result = np.empty((len(frame), len(OUTCOMES)), dtype=float)
        cache: dict[tuple, tuple[float, float, float, float]] = {}
        for index, row in enumerate(frame.loc[:, columns].itertuples(index=False, name=None)):
            # Prepared data contains categorical strings or None.  Normalize
            # here because CSV round trips can convert None into NaN.
            state = (row[0], row[1], *(None if pd.isna(value) else value for value in row[2:]))
            probabilities = cache.get(state)
            if probabilities is None:
                probabilities = self.predict_proba(row[0], row[1], state[2:])
                cache[state] = probabilities
            result[index] = probabilities
        return result

    def _smooth(self, observed: Iterable[int]) -> tuple[float, float, float, float]:
        counts = tuple(observed)
        denominator = sum(counts) + len(OUTCOMES) * self.alpha
        # fit() requires nonempty data, and this method is only called for
        # observed states or the nonempty global training counts.
        return tuple((count + self.alpha) / denominator for count in counts)  # type: ignore[return-value]

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Fit the CPT on training data before prediction")

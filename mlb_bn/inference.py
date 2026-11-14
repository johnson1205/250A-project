"""Inference for the simplified pitch-outcome Bayesian network.

The count and recent outcomes form the observed state.  The CPT supplies the
four outcome probabilities at each state.  ``transition`` then updates the
count and history deterministically, or ends the current at-bat.

Future-pitch probabilities are *conditional on the target pitch existing*.
Their companion ``existence_probability`` makes the conditioning explicit.
Pitches after a walk, strikeout, or ball in play are never generated.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .constants import OUTCOMES


class PitchModel(Protocol):
    history_length: int

    def predict_proba(
        self, balls: int, strikes: int, history: tuple[str | None, ...]
    ) -> tuple[float, float, float, float]: ...


@dataclass(frozen=True)
class State:
    """State immediately before a pitch; history is most-recent first."""

    balls: int
    strikes: int
    history: tuple[str | None, ...] = ()

    def __post_init__(self) -> None:
        if not 0 <= self.balls <= 3 or not 0 <= self.strikes <= 2:
            raise ValueError("A live at-bat needs 0 through 3 balls and 0 through 2 strikes")
        if not isinstance(self.history, tuple):
            raise TypeError("history must be a tuple, most-recent outcome first")
        if any(value is not None and value not in OUTCOMES for value in self.history):
            raise ValueError("history contains an unsupported outcome")


@dataclass(frozen=True)
class FutureResult:
    """Probability of Ball given pitch k exists, and probability it exists."""

    ball_probability: float | None
    existence_probability: float


@dataclass(frozen=True)
class AtBatResult:
    """Unconditional at-bat termination probabilities."""

    walk: float
    strikeout: float
    in_play: float
    unresolved: float = 0.0


def transition(state: State, outcome: str) -> tuple[State | None, str | None]:
    """Apply the simplified baseball rules after one observed outcome.

    A foul tip or foul bunt must be classified as Strike upstream; an ordinary
    Foul with two strikes leaves the count unchanged.  ``terminal`` is one of
    ``walk``, ``strikeout``, ``in_play`` or ``None``.
    """

    if outcome not in OUTCOMES:
        raise ValueError(f"unsupported outcome: {outcome!r}")
    if outcome == "InPlay":
        return None, "in_play"
    if outcome == "Ball" and state.balls == 3:
        return None, "walk"
    if outcome == "Strike" and state.strikes == 2:
        return None, "strikeout"

    balls = state.balls + (outcome == "Ball")
    strikes = state.strikes + (outcome == "Strike" or (outcome == "Foul" and state.strikes < 2))
    history = (outcome,) + state.history[:-1] if state.history else ()
    return State(balls, strikes, history), None


def _validate_query(model: PitchModel, state: State) -> None:
    if len(state.history) != model.history_length:
        raise ValueError("state history length must match model.history_length")


def _probabilities(
    model: PitchModel, state: State, cache: dict[State, tuple[float, ...]]
) -> tuple[float, ...]:
    if state not in cache:
        probabilities = tuple(float(p) for p in model.predict_proba(state.balls, state.strikes, state.history))
        if len(probabilities) != len(OUTCOMES) or any(not np.isfinite(p) or p < 0 for p in probabilities):
            raise ValueError("model must return four finite, nonnegative probabilities")
        if abs(sum(probabilities) - 1.0) > 1e-8:
            raise ValueError("model probabilities must sum to one")
        cache[state] = probabilities
    return cache[state]


def _future_result(ball_mass: float, existence_mass: float) -> FutureResult:
    return FutureResult(
        ball_probability=ball_mass / existence_mass if existence_mass > 0 else None,
        existence_probability=existence_mass,
    )


def future_ball_exact(model: PitchModel, state: State, horizon: int) -> FutureResult:
    """Eliminate the unknown intermediate outcomes using state propagation.

    ``horizon=1`` refers to the next pitch.  Terminal branches are removed
    before the target step, so the returned ball probability is conditional on
    that target pitch existing.  The survival mass is returned separately.
    """

    _validate_query(model, state)
    if horizon < 1:
        raise ValueError("horizon must be at least one")
    cache: dict[State, tuple[float, ...]] = {}
    active: dict[State, float] = {state: 1.0}
    for _ in range(horizon - 1):
        next_active: dict[State, float] = defaultdict(float)
        for current, mass in active.items():
            for outcome, probability in zip(OUTCOMES, _probabilities(model, current, cache)):
                if probability == 0:
                    continue
                successor, terminal = transition(current, outcome)
                if terminal is None:
                    assert successor is not None
                    next_active[successor] += mass * probability
        active = next_active
    existence_mass = sum(active.values())
    ball_mass = sum(mass * _probabilities(model, current, cache)[0] for current, mass in active.items())
    return _future_result(ball_mass, existence_mass)


def future_ball_enumeration(model: PitchModel, state: State, horizon: int) -> FutureResult:
    """Brute-force reference: enumerate every nonterminal outcome prefix."""

    _validate_query(model, state)
    if horizon < 1:
        raise ValueError("horizon must be at least one")
    cache: dict[State, tuple[float, ...]] = {}
    ball_mass = 0.0
    existence_mass = 0.0

    def visit(current: State, earlier_pitches: int, path_mass: float) -> None:
        nonlocal ball_mass, existence_mass
        if earlier_pitches == 0:
            existence_mass += path_mass
            ball_mass += path_mass * _probabilities(model, current, cache)[0]
            return
        for outcome, probability in zip(OUTCOMES, _probabilities(model, current, cache)):
            if probability == 0:
                continue
            successor, terminal = transition(current, outcome)
            if terminal is None:
                assert successor is not None
                visit(successor, earlier_pitches - 1, path_mass * probability)

    visit(state, horizon - 1, 1.0)
    return _future_result(ball_mass, existence_mass)


def future_ball_sample(
    model: PitchModel, state: State, horizon: int, n_samples: int, seed: int | None = None
) -> FutureResult:
    """Forward-sample at-bats to estimate the future-pitch query."""

    _validate_query(model, state)
    if horizon < 1 or n_samples < 1:
        raise ValueError("horizon and n_samples must be positive")
    rng = np.random.default_rng(seed)
    cache: dict[State, tuple[float, ...]] = {}
    existence_count = 0
    ball_count = 0
    for _ in range(n_samples):
        current = state
        for step in range(horizon):
            outcome = OUTCOMES[int(rng.choice(len(OUTCOMES), p=_probabilities(model, current, cache)))]
            if step == horizon - 1:
                existence_count += 1
                ball_count += outcome == "Ball"
                break
            successor, terminal = transition(current, outcome)
            if terminal is not None:
                break
            assert successor is not None
            current = successor
    return FutureResult(
        ball_probability=ball_count / existence_count if existence_count else None,
        existence_probability=existence_count / n_samples,
    )


def at_bat_exact(model: PitchModel, state: State) -> AtBatResult:
    """Solve the finite absorbing Markov chain for the current at-bat.

    States are grouped by count.  Balls and strikes only increase, except a
    two-strike foul, which stays at the same count.  Solving each count block
    therefore handles arbitrarily many fouls without truncating the at-bat.
    A closed nonterminating class makes the linear system singular; in that
    case, use positive smoothing or a CPT with nonzero terminal probability.
    """

    _validate_query(model, state)
    cache: dict[State, tuple[float, ...]] = {}
    reached: set[State] = {state}
    queue: deque[State] = deque([state])
    while queue:
        current = queue.popleft()
        for outcome, probability in zip(OUTCOMES, _probabilities(model, current, cache)):
            if probability == 0:
                continue
            successor, terminal = transition(current, outcome)
            if terminal is None and successor not in reached:
                assert successor is not None
                reached.add(successor)
                queue.append(successor)

    by_count: dict[tuple[int, int], list[State]] = defaultdict(list)
    for current in reached:
        by_count[current.balls, current.strikes].append(current)
    solved: dict[State, np.ndarray] = {}
    terminal_index = {"walk": 0, "strikeout": 1, "in_play": 2}
    for count in sorted(by_count, key=lambda item: sum(item), reverse=True):
        block = by_count[count]
        position = {current: i for i, current in enumerate(block)}
        matrix = np.eye(len(block), dtype=float)
        right = np.zeros((len(block), 3), dtype=float)
        for current, row in position.items():
            for outcome, probability in zip(OUTCOMES, _probabilities(model, current, cache)):
                if probability == 0:
                    continue
                successor, terminal = transition(current, outcome)
                if terminal is not None:
                    right[row, terminal_index[terminal]] += probability
                elif successor in position:
                    matrix[row, position[successor]] -= probability
                else:
                    assert successor is not None
                    right[row] += probability * solved[successor]
        try:
            result = np.linalg.solve(matrix, right)
        except np.linalg.LinAlgError as exc:
            raise ValueError("nonterminating at-bat state: use a CPT with positive terminal probability") from exc
        for current, row in position.items():
            solved[current] = result[row]

    walk, strikeout, in_play = (float(x) for x in solved[state])
    total = walk + strikeout + in_play
    if abs(total - 1.0) > 1e-7 or min(walk, strikeout, in_play) < -1e-9:
        raise ValueError("at-bat does not terminate with probability one")
    return AtBatResult(walk, strikeout, in_play)


def at_bat_sample(
    model: PitchModel,
    state: State,
    n_samples: int,
    seed: int | None = None,
    max_pitches: int = 10_000,
) -> AtBatResult:
    """Forward-sample at-bats; expose any runs that hit ``max_pitches``."""

    _validate_query(model, state)
    if n_samples < 1 or max_pitches < 1:
        raise ValueError("n_samples and max_pitches must be positive")
    rng = np.random.default_rng(seed)
    cache: dict[State, tuple[float, ...]] = {}
    counts = {"walk": 0, "strikeout": 0, "in_play": 0, "unresolved": 0}
    for _ in range(n_samples):
        current = state
        for _ in range(max_pitches):
            outcome = OUTCOMES[int(rng.choice(len(OUTCOMES), p=_probabilities(model, current, cache)))]
            successor, terminal = transition(current, outcome)
            if terminal is not None:
                counts[terminal] += 1
                break
            assert successor is not None
            current = successor
        else:
            counts["unresolved"] += 1
    return AtBatResult(*(counts[name] / n_samples for name in ("walk", "strikeout", "in_play", "unresolved")))

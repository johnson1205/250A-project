"""Checks of termination handling and exact inference against known answers."""

import unittest

from mlb_bn.inference import (
    State,
    at_bat_exact,
    at_bat_sample,
    future_ball_enumeration,
    future_ball_exact,
    future_ball_sample,
    transition,
)


class FixedModel:
    history_length = 0

    def __init__(self, probabilities):
        self.probabilities = probabilities

    def predict_proba(self, balls, strikes, history):
        return self.probabilities


class InferenceTests(unittest.TestCase):
    def test_count_transitions_and_two_strike_foul(self):
        self.assertEqual(transition(State(1, 1), "Foul"), (State(1, 2), None))
        self.assertEqual(transition(State(1, 2), "Foul"), (State(1, 2), None))
        self.assertEqual(transition(State(3, 1), "Ball"), (None, "walk"))
        self.assertEqual(transition(State(0, 2), "Strike"), (None, "strikeout"))
        self.assertEqual(transition(State(0, 0), "InPlay"), (None, "in_play"))

    def test_history_updates_only_after_observed_pitch(self):
        current = State(0, 0, ("Strike", "Ball", None))
        self.assertEqual(
            transition(current, "Foul"),
            (State(0, 1, ("Foul", "Strike", "Ball")), None),
        )

    def test_future_pitch_conditioning_and_enumeration(self):
        model = FixedModel((0.2, 0.3, 0.4, 0.1))
        state = State(3, 2)
        for horizon in range(1, 5):
            exact = future_ball_exact(model, state, horizon)
            enumerated = future_ball_enumeration(model, state, horizon)
            self.assertAlmostEqual(exact.ball_probability, 0.2)
            self.assertAlmostEqual(exact.existence_probability, 0.4 ** (horizon - 1))
            self.assertAlmostEqual(exact.ball_probability, enumerated.ball_probability)
            self.assertAlmostEqual(exact.existence_probability, enumerated.existence_probability)

    def test_absorbing_chain_sums_all_two_strike_fouls(self):
        model = FixedModel((0.2, 0.3, 0.4, 0.1))
        result = at_bat_exact(model, State(3, 2))
        self.assertAlmostEqual(result.walk, 0.2 / 0.6)
        self.assertAlmostEqual(result.strikeout, 0.3 / 0.6)
        self.assertAlmostEqual(result.in_play, 0.1 / 0.6)

    def test_forward_sampling_agrees_with_exact(self):
        model = FixedModel((0.2, 0.3, 0.4, 0.1))
        state = State(3, 2)
        future = future_ball_sample(model, state, horizon=3, n_samples=20_000, seed=2)
        self.assertAlmostEqual(future.ball_probability, 0.2, delta=0.025)
        self.assertAlmostEqual(future.existence_probability, 0.16, delta=0.01)
        at_bat = at_bat_sample(model, state, n_samples=20_000, seed=2)
        self.assertAlmostEqual(at_bat.walk, 1 / 3, delta=0.02)
        self.assertEqual(at_bat.unresolved, 0)


if __name__ == "__main__":
    unittest.main()

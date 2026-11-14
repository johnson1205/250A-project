"""Small checks for CPT estimation and training-only backoff."""

import unittest

import numpy as np
import pandas as pd

from mlb_bn.model import CPTModel


class CPTModelTests(unittest.TestCase):
    def test_laplace_probabilities_and_binary_baseline(self) -> None:
        train = pd.DataFrame(
            {
                "balls": [0, 0, 1],
                "strikes": [0, 0, 0],
                "outcome": ["Ball", "Strike", "Foul"],
            }
        )
        model = CPTModel(alpha=1).fit(train)
        self.assertTrue(np.allclose(model.predict_proba(0, 0), (2 / 6, 2 / 6, 1 / 6, 1 / 6)))
        self.assertEqual(model.majority_class, "NotBall")
        self.assertAlmostEqual(model.class_frequency["Ball"], 1 / 3)
        self.assertAlmostEqual(sum(model.predict_proba(0, 0)), 1)

    def test_unseen_history_backs_off_to_training_count(self) -> None:
        train = pd.DataFrame(
            {
                "balls": [0, 0, 1],
                "strikes": [0, 0, 0],
                "outcome": ["Ball", "Strike", "Foul"],
                "h1_carry": [None, "Ball", "Strike"],
                "h2_carry": [None, None, "Ball"],
            }
        )
        model = CPTModel(history_length=2, alpha=1).fit(train)
        expected_count = (2 / 6, 2 / 6, 1 / 6, 1 / 6)
        self.assertTrue(np.allclose(model.predict_proba(0, 0, ("Foul", "Ball")), expected_count))
        # This count never occurred in training, so it uses global frequencies.
        self.assertTrue(np.allclose(model.predict_proba(3, 2, ("Foul", "Ball")), (2 / 7, 2 / 7, 2 / 7, 1 / 7)))

    def test_zero_alpha_is_mle_and_prediction_rows_keep_order(self) -> None:
        train = pd.DataFrame(
            {
                "balls": [0, 0, 1],
                "strikes": [0, 0, 0],
                "outcome": ["Ball", "Ball", "Strike"],
                "h1_reset": [None, "Ball", None],
            }
        )
        model = CPTModel(history_length=1, strategy="reset", alpha=0).fit(train)
        self.assertEqual(model.predict_proba(0, 0, ("Ball",)), (1.0, 0.0, 0.0, 0.0))
        check = pd.DataFrame({"balls": [1, 0], "strikes": [0, 0], "h1_reset": [None, np.nan]})
        probabilities = model.predict_frame(check)
        self.assertEqual(probabilities.shape, (2, 4))
        self.assertTrue(np.allclose(probabilities[0], (0, 1, 0, 0)))
        self.assertTrue(np.allclose(probabilities[1], (1, 0, 0, 0)))


if __name__ == "__main__":
    unittest.main()

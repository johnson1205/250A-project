# Full-season experiment results

The run used regular-season Statcast pitches from 2023 for training, 2024 for validation, and 2025 for testing. The selected model and smoothing value were chosen by 2024 binary Ball log-loss. No 2025 outcomes were used to fit or select the model.

| Season | Role | Modeled pitches | Excluded descriptions |
| --- | --- | ---: | ---: |
| 2023 | Train | 715,833 | 4,851 |
| 2024 | Validate | 707,492 | 4,407 |
| 2025 | Test | 708,156 | 4,372 |

The excluded descriptions were `automatic_ball`, `automatic_strike`, and `hit_by_pitch`. See `data_report.json` for counts by season.

## Next-pitch prediction

The 2024 validation set selected **count + last two pitches with history carried across at-bats**, using Laplace **alpha = 1**. Its validation log-loss was **0.644198**.

| 2025 model | Binary log-loss | Ball F1 | 10-bin ECE |
| --- | ---: | ---: | ---: |
| Majority probability baseline | 0.652165 | 0 | 0.000672 |
| Count only | 0.646081 | 0 | 0.002803 |
| Selected two-pitch carry | **0.645932** | 0.000047 | 0.002162 |
| Three-pitch carry | 0.645984 | 0 | 0.001438 |

The selected model improves binary log-loss over count only by **0.000149 per pitch**. This is a small predictive gain. The three-pitch carry model does not improve on the selected two-pitch model in this test. Full comparisons are in `test_metrics.csv`.

At a fixed 0.5 probability threshold, nearly every model predicts `NotBall`; consequently Ball-class F1 is almost zero and does not distinguish the models well. The training Ball rate is **0.35851**. Log-loss and the calibration table are more informative for these probabilistic predictions. The threshold was fixed before examining 2025 results.

In the three-pitch carry CPT at a 2-1 count, `Ball-Ball-Strike` gives **P(Ball)=0.30065** and `Strike-Ball-Ball` gives **P(Ball)=0.28976**. This is a descriptive comparison of learned conditional probabilities, not a causal claim.

## Future-pitch and at-bat inference

The query starts at a 0-0 count with no previous pitch history in the selected model. The future Ball probabilities below are conditional on that future pitch existing; at-bats that end earlier are excluded from the conditional probability.

| Pitch horizon | P(target pitch exists) | P(Ball given target exists) |
| ---: | ---: | ---: |
| 1 | 1.00000 | 0.37533 |
| 2 | 0.89335 | 0.37514 |
| 3 | 0.74574 | 0.37088 |
| 4 | 0.56737 | 0.35428 |

Exact at-bat inference gives **P(walk)=0.08385**, **P(strikeout)=0.23305**, and **P(ball in play)=0.68310**. Exact state propagation matched brute-force enumeration for horizons 1–4. Among the tested sample sizes (100, 1,000, 10,000), 1,000 was the smallest with mean absolute error at most 0.02 across five seeded repetitions for each reported future Ball, walk, and strikeout query. That sample-size result is descriptive and not a guarantee.

The per-pitch predictions, full metrics, calibration bins, and inference timings are in the CSV files beside this summary.

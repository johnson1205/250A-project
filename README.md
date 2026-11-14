# Predicting Ball vs. Strike from Pitch History Using Bayesian Networks

This course project tests whether a pitcher's recent outcomes improve next-pitch
prediction after the current ball-strike count is known. It uses one discrete
conditional probability table (CPT) per model. The four outcomes are `Ball`,
`Strike`, `Foul`, and `InPlay`; the binary target is `Ball` versus `NotBall`
(`Strike` + `Foul` + `InPlay`).

## Run the study

Use Python 3.11 in the project directory:

```bash
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install -r requirements.txt
python -m pytest -q
python -m mlb_bn.experiment --quick
```

`--quick` uses April 1–7 in each of 2023, 2024, and 2025. The complete study is:

```bash
python -m mlb_bn.experiment
```

The full run fetches March 1–November 15 of each year and keeps only
`game_type == "R"` (regular-season) records. The broad date range covers the
regular season while filtering out other game types. It can take substantial
time and disk space. Fetches are split into seven-day, year-specific chunks in
`data/cache/<year>/`; successfully fetched chunks are reused on later runs,
including when a quick run is followed by a full run. A failed fetch raises an
error and does not create a fake result. You can change the locations with
`--cache-dir` and `--output-dir`, and the chunk length with `--chunk-days`.

The default smoothing candidates are `0.1`, `1`, and `10`. To try others:

```bash
python -m mlb_bn.experiment --quick --alphas 0.1 0.5 1 2 5
```

The default seed for sampling is `250`; change it with `--seed`.

## Data and pitch history

The loader uses `game_pk`, `game_date`, `pitcher`, `at_bat_number`,
`pitch_number`, `balls`, `strikes`, `description`, and `game_type` from
[Baseball Savant's Statcast CSV](https://baseballsavant.mlb.com/csv-docs), via
[pybaseball's `statcast` function](https://github.com/jldbc/pybaseball/blob/master/docs/statcast.md).
It saves only these and a few optional fields in its compressed CSV cache.
Within each game it orders pitches by at-bat and pitch number before making
history. The outcome mapping is explicit in `mlb_bn/data.py`.

Ordinary fouls are `Foul`. Foul tips and foul bunts are `Strike`, because they
can be strike three ([foul tip](https://www.mlb.com/glossary/rules/foul-tip),
[foul bunt](https://www.mlb.com/official-information/umpires/strike-zone)).
In-play variants are `InPlay`. Hit by pitch, pitch-clock
automatic calls, unknown descriptions, and records with invalid pre-pitch
counts are excluded. An excluded row clears the history buffers, so valid
pitches on either side are never treated as consecutive. A missing pitch-number
or at-bat-number gap also clears carried history. `data_report.json` records
the number excluded by reason and by unsupported description. No unknown
description is silently assigned to `Strike`.

`h1` means the immediately previous observed pitch, `h2` the one before, and
`h3` the third one back. Each missing slot is `None`. In the main `carry`
definition, history continues across at-bats while the same pitcher remains
the current pitcher in that game. It clears on a new game or pitching change.
In the `reset` definition, it also clears at the start of each at-bat.

For example, before the current pitch, with histories listed most recent first:

| Pitch | Observed outcome | Carry history before pitch | Reset history before pitch |
| --- | --- | --- | --- |
| At-bat 1, pitch 1 | Ball | `(None, None, None)` | `(None, None, None)` |
| At-bat 1, pitch 2 | Strike | `(Ball, None, None)` | `(Ball, None, None)` |
| At-bat 1, pitch 3 | InPlay | `(Strike, Ball, None)` | `(Strike, Ball, None)` |
| At-bat 2, pitch 1 | Foul | `(InPlay, Strike, Ball)` | `(None, None, None)` |
| At-bat 2, pitch 2 | Ball | `(Foul, InPlay, Strike)` | `(Foul, None, None)` |
| New pitcher, first pitch | Ball | `(None, None, None)` | `(None, None, None)` |

This is an illustrative sequence, not a Statcast result.

## CPT learning and model comparison

For each count and history state, the training frequency of each outcome is
converted to a probability with

`P(o | b,s,h) = (n(b,s,h,o) + alpha) / (n(b,s,h) + 4*alpha)`.

With `alpha=0`, this is maximum likelihood estimation for observed states.
For a previously unseen full state, prediction drops the oldest history value
until it finds a state observed in training, then uses count alone, then the
overall 2023 outcome frequencies. Positive `alpha` smooths each distribution.
This backoff and every fitted count use 2023 data only.

The seven CPT comparisons are count only (`history_length=0`), and history
lengths 1, 2, and 3 with both `carry` and `reset`. Each model chooses its
Laplace `alpha` by lowest **binary Ball log-loss on 2024**. The overall model
and history strategy are also selected by 2024 log-loss; 2025 is used once for
final evaluation. A separate majority-class baseline always predicts the
most common 2023 binary class as its hard label, while its probability is the
2023 Ball frequency. Ties in the hard label resolve to `NotBall`.

The 2025 report includes binary log-loss, Ball-class F1, precision, recall,
accuracy, Brier score, and calibration error from ten equal-width bins.
`test_calibration.csv` gives each bin's predicted and observed Ball rate.
Four-way log-loss is also included for the CPTs. A binary hard prediction uses
`p_ball >= 0.5` for CPTs. The two ordered paths to a 2-1 count from the
proposal are compared in `history_interpretation.csv` with the three-pitch
carry model.

## Inference within an at-bat

The CPT is the stochastic node `Outcome_t`. The pre-pitch count and recent
history are observed parents; the count and history after the pitch are
deterministic children of the prior state and outcome. A fourth `Ball` ends in
a walk, a third `Strike` in a strikeout, and `InPlay` ends the at-bat. `Foul`
adds a strike only when there are fewer than two strikes. The model does not
simulate any pitch after the at-bat ends.

`future_ball_exact` merges identical states while summing over unobserved
intermediate pitch outcomes, equivalent to eliminating those outcome variables.
`future_ball_enumeration` visits each outcome path as a check. For horizons
1–4, both report **P(Ball at pitch k | pitch k exists)** and the probability
that pitch k exists. If survival probability is zero, the conditional Ball
probability is undefined and reported as missing. The `at_bat_exact` function
solves the finite absorbing-chain equations for walk, strikeout, and ball in
play, including arbitrarily many two-strike fouls. Unsmoothened CPTs can have
nonterminating closed states; in that case it raises an error rather than
inventing an at-bat result. The default positive smoothing avoids this.

Forward sampling estimates the same quantities using simulated at-bats.
The experiment compares exact inference, enumeration, and five seeded
sampling repetitions at 100, 1,000, and 10,000 samples. It records runtime and
absolute error. `sampling_requirement.csv` gives the smallest **tested** sample
size with mean absolute error at most 0.02 across those five repetitions;
this is a descriptive observation, not a statistical guarantee.

## Output files

`results/` is created when a run succeeds:

| File | Contents |
| --- | --- |
| `data_report.json` | Per-season row counts and exclusions |
| `run_config.json` | Split, candidate alphas, chosen model, seed |
| `validation_metrics.csv` | Every model and alpha evaluated on 2024 |
| `test_metrics.csv` | Majority baseline and seven CPTs evaluated on 2025 |
| `test_calibration.csv` | Ten-bin calibration for every test model |
| `test_predictions.csv` | Four outcome probabilities and Ball/NotBall prediction for every 2025 pitch, using the 2024-selected model |
| `history_interpretation.csv` | The two ordered 2-1 count examples |
| `future_inference.csv` | Exact, enumerated, and sampled horizon 1–4 queries |
| `at_bat_inference.csv` | Walk, strikeout, and in-play queries |
| `sampling_summary.csv` | Sampling error and time by query and sample size |
| `sampling_requirement.csv` | Smallest tested sample count reaching the stated mean-error threshold |

No downloaded Statcast data or empirical findings are included in this
repository. The tests use small, explicitly constructed fixtures.

After an experiment run, create PNG and PDF figures with:

```bash
python -m mlb_bn.plot_results --results-dir results/full
```

The figures are saved in `results/full/figures/`.

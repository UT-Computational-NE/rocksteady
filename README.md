# rocksteady

A reusable pattern for deciding when an experiment is in steady state, with
adapters for two systems that look different on the surface: the UT TRIGA
reactor (power and control rods) and the STNCL natural-circulation loop
(sixteen thermocouples and derived temperatures).

## The pattern

Both existing detectors turned out to be the same machine with different
parts plugged in. A detector is assembled from five kinds of parts:

| Part | What it is | TRIGA | STNCL |
|---|---|---|---|
| **Signals** | Named scalars computed from each raw sample | power (zeroed below 1 W), 4 rods | 16 thermocouples, `T_avg`, spreads, offset, `T_diag` |
| **Criteria** | A test on one signal over the trailing window, returning a statistic, its uncertainty, and a limit | `DeviationCriterion`: max deviation from a constant fit (power) or a linear fit (rods), with a measurement-noise model | `DriftCriterion`: slope equivalence `|b| + 2u_b <= b_crit`, Newey-West `u_b` |
| **Disturbances** | External action, detected in the signals or supplied from a log | `StepChange` on each rod | `EventLog` of control and configuration events |
| **Timing** | Block length, window, settling time after a disturbance, hold time, undecided policy | 30 s blocks, 300 s window (about 4 delayed-neutron time constants), no hold, hold last decision when undecided | 30 s blocks, 20 or 45 min window, settle `tau_slow,op`, hold 20 or 45 min, undecided is not accepted |
| **Overrides** | Domain rules that force a label | zero-power rule | none yet (the energy-residual criterion would go here or as a criterion) |

The shared state machine then produces one label per sample:

| Label | Meaning |
|---|---|
| `disturbed` | A disturbance is active now (TRIGA: `active`) |
| `settling` | Within the quiet time after a disturbance, or a criterion fails beyond its uncertainty |
| `indeterminate` | Not enough history, or the uncertainty straddles a limit and no earlier decision is held |
| `holding` | All criteria pass, but for less than the hold time |
| `steady` | All criteria pass and the hold time has elapsed |

Everything except `steady` and `holding` is transient in the two-class sense.

Two ideas make the pattern hold across both systems:

1. **Every criterion reports uncertainty.** A criterion returns
   `(statistic, uncertainty, limit, k)`. The detector forms the margin
   ratio `(statistic + k * uncertainty) / limit` (the loop plan's `eta`)
   and a three-way test: pass if the upper bound is within the limit, fail
   if the lower bound exceeds it, undecided otherwise. The reactor's
   noise-model bounds and the loop's HAC slope uncertainty are two ways of
   supplying that uncertainty. The linear-rod exemption is just a
   different fit model inside one criterion type.
2. **Disturbances reset clocks; the window certifies settling.** Whether a
   disturbance is found in the signals (a rod move) or read from a log (a
   valve change), the response is the same: the state goes to `disturbed`,
   then `settling` until the window plus any settling precondition has
   elapsed without another disturbance, and only then are the criteria
   evaluated. The window length is where each system's time constant
   enters.

The detector is causal. Running it over a stored record (`batch.run`)
gives exactly the labels the online detector would have produced, which is
what makes archived labels and control-room labels comparable.

## Validation

- **TRIGA**: `adapters/triga_steady_state.py` reproduces the labels of the
  existing `steady_state_classifier.OnlineSteadyStateTagger` sample for
  sample on 2026-01-12 and 2026-01-15 (172,802 samples, zero mismatches).
  `examples/validate_triga.py`.
- **STNCL**: `adapters/stncl_steady_state.py` reproduces
  `loop_steady_state.py` on `run7_02_26.csv`: margin ratios agree to
  1e-13, pass/fail agrees on all 441 (operating) and 391 (calibration)
  block-aligned windows, and with limits relaxed so that windows do pass,
  297 of 297 agree with the same accepted spans.
  `examples/validate_stncl.py`.

One convention differs: the reference loop script stamps each window at
its centre (post-run plotting), the library at its end (the time the state
is confirmed). For a 20-minute window the reported start times differ by
10 minutes.

## Using it

```python
import rocksteady as rs

signals = [rs.Signal("x", "x"), rs.Signal("mean_ab", lambda s: (s["a"] + s["b"]) / 2)]
criteria = [
    rs.DeviationCriterion("x", limit=0.5, noise=0.05, model="constant"),
    rs.DriftCriterion("mean_ab", limit_per_s=0.01 / 60),
]
disturbances = [rs.StepChange("x", threshold=2.0, m=10)]
timing = rs.Timing(sample_period_s=1.0, block_s=30.0, window_s=300.0, hold_s=0.0)

det = rs.Detector(signals, criteria, disturbances, timing=timing)

# online
verdict = det.update(t_seconds, {"x": 1.0, "a": 2.0, "b": 3.0})
verdict.label, verdict.eta, verdict.limiting

# batch
labels = rs.batch.run(det, dataframe)          # one row per sample
rs.batch.accepted_intervals(labels["label"])   # [start, stop) index runs
```

Post-acceptance reporting (mean with an autocorrelation-aware
uncertainty, the loop plan's `N_eff`) is in `rs.stats.mean_with_uncertainty`.

### Extending

- A new test on a signal: subclass `Criterion` and return a
  `CriterionResult`. Examples that would fit: the loop's energy-residual
  criterion once secondary duties are logged; a reactor period limit.
- A new source of disturbance: subclass `Disturbance`. The loop plan's
  optional Cao-Rhinehart local-excursion veto is a `Disturbance` whose
  `update` returns True on an excursion.
- A new domain rule: subclass `Override`. It sees every sample (so it can
  keep state, as the zero-power rule does) and may force a label; a
  disturbance always takes precedence.

## How to deploy it in both projects

Make `rocksteady` its own repository and install it into each project as
a pinned dependency. Keep the adapters in the projects, not in the library:

```
rocksteady/            (this package: generic, no reactor or loop knowledge)
triga_modsim_tools/dataproc/triga_steady_state.py   (from adapters/)
stncl_analysis/stncl_steady_state.py                 (from adapters/)
```

In each project's `pyproject.toml` (or requirements file):

```
rocksteady @ git+ssh://git@github.com/<org>/rocksteady.git@v0.1.0
```

Pin a tag, and bump it deliberately. Each project should keep its
validation script (`examples/validate_*.py`) as a regression test, so a
library change that alters labels is caught before it reaches the archive.

Adapter responsibilities, so the split stays clean:

- **Library**: the state machine, the criteria and disturbance types, the
  statistics, batch and interval utilities. Nothing in it knows about
  rods, thermocouples, or acceptance classes.
- **Adapter**: signal definitions, calibrated tolerances and noise models,
  acceptance classes, settling constants, domain overrides, and any legacy
  interface (`triga_steady_state.OnlineSteadyStateTagger` keeps the
  `update(power, rods) -> str` API, so `data_tagger.py` only changes its
  import).

Vendoring (copying `rocksteady/` into each project) also works and avoids
a dependency, at the cost of the two copies drifting apart. The pinned git
dependency is the better default.

## Layout

```
rocksteady/
  core.py            Signal, Criterion, Disturbance, Override, Timing, Detector, Verdict
  criteria.py        DeviationCriterion, DriftCriterion, BoundCriterion
  disturbances.py    StepChange, EventLog
  stats.py           block means, HAC slope, tau_int, mean_with_uncertainty
  batch.py           run, intervals, accepted_intervals
adapters/
  triga_steady_state.py
  stncl_steady_state.py
examples/
  validate_triga.py  validate_stncl.py
tests/
  test_core.py       synthetic tests of the state machine and criteria
```

Run the tests with `python -m pytest tests`.

## Performance

`batch.run` is a Python loop over samples: about 8 s per reactor day
(86,400 samples, 5 criteria) and about 17 s per four-hour loop run
(14,400 samples, 21 HAC fits per sample). The previous reactor classifier
was about twice as fast; the difference is the generic bookkeeping. If it
matters, evaluate criteria only at block boundaries (the loop reference
does this) by adding an evaluation stride to `Timing`.

# rocksteady

Steady-state detection for experiments and processes with several signals,
characteristic time constants, and external actions.

`rocksteady` labels every sample of a record, or every sample of a live
stream, as `disturbed`, `settling`, `indeterminate`, `holding` or `steady`.
It accepts a steady state only when the data show it: each test reports a
statistic with its uncertainty, so a window is accepted only when the
whole uncertainty interval is inside the limit. The detector is causal. Running it over a stored
record gives exactly the labels it would have given online.

Change-point packages such as
[ruptures](https://github.com/deepcharles/ruptures) find where a signal
changes, after the fact. `rocksteady` answers a different question,
continuously: *has the system settled, and can that be demonstrated from
the data?*

Documentation: https://rocksteady.readthedocs.io

## Install

```
pip install "rocksteady @ git+https://github.com/UT-Computational-NE/rocksteady.git@v0.5.0"
```

Python 3.10 or later, with numpy and pandas; both are installed with the
package. The streaming detector (`Detector.update` and its components)
needs numpy alone and does not import pandas. The batch, statistics and
report layers (`batch`, `metrics`, `report`) return DataFrames and use
pandas.

## Quickstart

```python
import rocksteady as rs
from rocksteady.signals import mean, spread

channels = ["T1", "T2", "T3", "T4"]
signals = [rs.Signal(c, c) for c in channels]
signals += [rs.Signal("T_mean", mean(channels)), rs.Signal("T_spread", spread(channels))]

detector = rs.Detector(
    signals,
    criteria=[rs.DriftCriterion(s.name, limit_per_s=0.01 / 60) for s in signals],  # 0.01 K/min
    disturbances=[rs.EventLog([(7200.0, 0.0)])],  # a setpoint change from the operator log
    timing=rs.Timing(sample_period_s=1.0, block_s=30.0, window_s=1200.0, settle_s=600.0, hold_s=1200.0),
)

# Online: one sample at a time.
verdict = detector.update(t_seconds, {"T1": 60.1, "T2": 60.4, "T3": 59.9, "T4": 60.7})
verdict.label, verdict.eta, verdict.limiting

# Offline: a whole record (one row per sample).
labels = rs.batch.run(detector, df)
rs.batch.accepted_intervals(labels["label"])  # [start, stop) index runs that reached steady
```

The same detector, as plain data that can be stored with a run as JSON,
YAML or TOML:

```python
spec = {
    "timing": {"sample_period_s": 1.0, "block_s": 30.0, "window_s": 1200.0,
               "settle_s": 600.0, "hold_s": 1200.0},
    "signals": ["T1", "T2", "T3", "T4",
                {"name": "T_mean", "builder": "mean", "cols": ["T1", "T2", "T3", "T4"]},
                {"name": "T_spread", "builder": "spread", "cols": ["T1", "T2", "T3", "T4"]}],
    "criteria": [{"type": "drift", "signal": s, "limit_per_s": 0.01 / 60}
                 for s in ["T1", "T2", "T3", "T4", "T_mean", "T_spread"]],
    "disturbances": [{"type": "event_log", "events": [[7200.0, 0.0]]}],
}
detector = rs.build_detector(spec)
```

Runnable examples with synthetic data are in [`examples/`](https://github.com/UT-Computational-NE/rocksteady/tree/main/examples):

- [`thermal_drift.py`](https://github.com/UT-Computational-NE/rocksteady/blob/main/examples/thermal_drift.py): channels relaxing to a setpoint, with an event log, settling time and hold time. It then fits the approach to the new setpoint with a first-order model (τ recovered as 600 ± 1 s) and compares two channels by state.
- [`actuator_steps.py`](https://github.com/UT-Computational-NE/rocksteady/blob/main/examples/actuator_steps.py): a level held by a stepping actuator, with a level-dependent noise model and a shutdown override. It ends with a fit report by state.

## Concepts

A detector is built from five kinds of parts. The state machine that
combines them is shared.

| Part | What it is | Built in |
|---|---|---|
| **Signals** | Named scalars computed from each raw sample: a column, or a derived quantity | column, `mean`, `spread`, `difference` |
| **Criteria** | Tests on one signal over the trailing window of block means. Each returns a statistic, its uncertainty, and a limit | `DriftCriterion` (no resolvable slope), `DeviationCriterion` (stays within a limit of a constant or a line), `BoundCriterion` (mean inside a band), `AsymptoteCriterion` (a first-order approach is within a limit of its final value) |
| **Disturbances** | External action, found in the signals or read from a log; resets the clocks | `StepChange` (a discrete move in a signal), `EventLog` (logged events with durations) |
| **Timing** | Sample spacing, block length, window, settling time after a disturbance, hold time, gap tolerance, and what an undecided window means. Block and window lengths are required: `rs.suggest_timing` proposes them from your data | `Timing` |
| **Overrides** | Domain knowledge that forces a label | `FloorLatch` (below a floor the system is steady by definition, for example when shut down) |

### Labels

| Label | Meaning |
|---|---|
| `disturbed` | A disturbance is active now |
| `settling` | Within the quiet time after a disturbance (window plus `settle_s`), or a criterion fails beyond its uncertainty |
| `indeterminate` | Not enough history, missing data, or an uncertainty that straddles a limit with no earlier decision held |
| `holding` | Every criterion passes, but for less than `hold_s` |
| `steady` | Every criterion passes and `hold_s` has elapsed |

In a two-class view, `steady` and `holding` are steady and everything else is transient.

### Margin ratio and the three-way test

Each criterion reports `(statistic, uncertainty, limit, k)`. The true value
is taken to lie within `statistic ± k·uncertainty`, so:

- **pass** when `statistic + k·uncertainty <= limit`;
- **fail** when `statistic - k·uncertainty > limit`;
- **undecided** otherwise.

The margin ratio `(statistic + k·uncertainty) / limit` is below 1 for a
pass. `Verdict.eta` is the worst ratio over the criteria, and
`Verdict.limiting` names the criterion that has it.

A disturbance resets the clocks, whether it was found in a signal or read
from a log. The state goes to `disturbed`, then `settling` until a full
window plus `settle_s` has passed without another disturbance, and only
then are the criteria evaluated. The window length is where the system's
time constant enters.

## Drift tests and autocorrelation

`DriftCriterion` passes when `|b| + k·u_b <= limit`, where `b` is the OLS
slope of the window's block means. That is an equivalence test: it
certifies that any drift is smaller than the limit. Block means of real
signals are usually autocorrelated, and an uncertainty that ignores this
certifies limits the data do not support.

The slope uncertainty is pluggable (`method=`):

- **`"ar1"` (default):** an AR(1) error model. The residual autocorrelation is bias-corrected, the slope variance is exact for that model, and `k` is a Student-t quantile on the window's effective degrees of freedom.
- **`"nw"`:** Newey–West with a data-dependent bandwidth and `k = 2`. This was the method of version 0.1.

The table gives Monte Carlo false-pass rates at the limit, for AR(1) noise on the
block means. The nominal rate is 2.5 % for `"ar1"` and 2.3 % for `"nw"`. Reproduce it with
`python tools/coverage_mc.py`:

| blocks in window | 10 | | | | 40 | | | | 90 | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lag-1 autocorrelation φ | 0 | 0.5 | 0.8 | 0.9 | 0 | 0.5 | 0.8 | 0.9 | 0 | 0.5 | 0.8 | 0.9 |
| `"ar1"` | 1.7 | 5.1 | 7.7 | 9.3 | 1.9 | 3.9 | 4.6 | 4.8 | 2.4 | 3.0 | 3.3 | 4.1 |
| `"nw"` | 7.2 | 16.0 | 24.6 | 27.9 | 3.6 | 10.6 | 17.7 | 23.5 | 2.8 | 7.3 | 12.9 | 18.2 |

Guidance:

- **Window length:** use at least about 40 blocks per window. With 10 blocks, no method here is reliable.
- **Strong autocorrelation:** when φ is high, `"ar1"` makes `k` large (up to 12.7). A window then certifies only a limit well above its apparent drift. With φ = 0.9, a 40-block window holds about two independent samples, so this is the honest answer. Longer blocks lower φ.
- **Your own method:** register it in `rocksteady.registry.slope_methods`. The harness calibrates every registered method.

### Slow approaches: the asymptote test

During a slow approach to a final value (a thermal relaxation, for
example), the residuals about a line stay strongly correlated, and the drift
test certifies little even with generous limits. `AsymptoteCriterion` tests
the approach itself. It certifies that the **remaining approach**
`d = A·exp(−t_end/τ)` of a first-order curve `y∞ + A·exp(−t/τ)` is within a
limit:

```python
rs.AsymptoteCriterion("T_avg", limit=0.05, tau_max=3000.0)   # 0.05 K left, tau at most 50 min
```

`τ` is rarely well determined by one window, and trusting a fitted `τ`
passes too often: noise that makes the curve look sharper makes `τ` look
shorter. So the test takes the **worst case over every `τ` up to `tau_max`
that the data allow** (a profile-likelihood confidence set), each with its
AR(1) uncertainty. `tau_max` is your upper bound on the process's time
constant. It makes every window decidable: a flat window, or one too short
to resolve the curve, is bounded by about `tau_max × |slope|`.

Monte Carlo false-pass rates at the limit (the true remaining approach equal
to the limit, AR(1) block-mean noise at 0.1 × the limit, `tau_max` = 2 τ),
for windows spanning 0.5 to 8 τ; nominal 2.5 %. Reproduce it with
`python tools/coverage_mc.py --asymptote`:

| blocks, φ | 0.5 τ | 1 τ | 2 τ | 4 τ | 8 τ |
|---|---|---|---|---|---|
| 20, 0 / 0.5 / 0.8 | 2.0 / 3.2 / 5.1 | 1.4 / 3.0 / 4.7 | 0.8 / 1.9 / 3.9 | 0.0 / 0.1 / 0.1 | 0 |
| 40, 0 / 0.5 / 0.8 | 1.5 / 2.4 / 2.5 | 1.1 / 2.1 / 2.5 | 0.5 / 1.2 / 1.4 | 0 | 0 |
| 90, 0 / 0.5 / 0.8 | 1.6 / 1.9 / 2.0 | 1.0 / 1.5 / 1.6 | 0.5 / 0.5 / 1.0 | 0 | 0 |

Give `tau_max` some margin. With `tau_max` equal to the true τ, the rates
rise to at most 6 % at 40 blocks. Power grows with the window's span and
the number of blocks. A window with a quarter of the limit left passes 96 %
of the time at 40 blocks spanning one τ (φ = 0), and 58 % at φ = 0.5.

## Gaps and timestamps

Times passed to `update` must increase and lie on a grid of
`sample_period_s`. Each step must be within `Timing.grid_tolerance` sample
spacings of a whole number of spacings. The default, 0.5, rounds any step of
at least half a spacing to the grid; set it lower (for example 0.1) to
reject jittery or drifting clocks. A record sampled faster than
`sample_period_s` is rejected. Resample it first, for example with
`df.resample("1s").mean()`, which also leaves NaN rows where samples are
missing.

By default (`Timing.max_gap_s = 0`),
the detector clears its history on any of these:

- a non-finite sample;
- an unreadable sample;
- a step in time of more than 1.5 sample spacings.

With `max_gap_s > 0`, shorter gaps are kept as missing data. Block means
then ignore the missing samples, and a window is evaluated only if every
block keeps at least `min_block_coverage` of its samples.
`rocksteady.batch.run` takes sample times from a `DatetimeIndex` when there is one.

The output of `batch.run` has one row per input row, with the same index:

| Column | Meaning |
|---|---|
| `label`, `reason` | the label and why it was given (below) |
| `eta`, `limiting` | the worst margin ratio and its criterion |
| `since_disturbance_s` | time since the last disturbance |
| `forced_by` | the override that forced the label, if any |
| `valid` | the sample was readable and finite |
| `window_start_s` | start of the window examined on this row (for a passing row, its certifying window), on the `t_s` clock |
| `ratio_<criterion>` | with `with_results=True` |
| `t_s` | sample time in seconds |

`segments_from`, `batch.fit_report` and the statistics use `t_s`. Backfill
reaches the certifying window by time, and autocorrelation is never
estimated across omitted timestamps.

A tolerated missing sample is labelled `indeterminate` whatever the
undecided policy, because no window is evaluated on it. The hold clock
**pauses** over it: it neither resets nor advances, for NaN rows and
omitted timestamps alike. `segments_from` bridges such samples inside a
steady run, so a gap given as NaN rows and the same gap given as omitted
timestamps produce the same steady segments.

### Why a sample has its label

Every `Verdict` has a `reason` (`rs.Reason`), and `batch.run` returns it as a
`reason` column:

| Reason | Meaning |
|---|---|
| `certified` | all criteria pass on the window |
| `forced` | an override forced the label (`forced_by` names it) |
| `disturbance` | a disturbance is active |
| `settling_time` | the quiet time after a disturbance (or at the start of a record) has not elapsed |
| `failed` | a criterion fails beyond its uncertainty |
| `undecided` | statistical overlap: the uncertainty straddles a limit |
| `invalid_result` | a criterion returned an unusable result (`CriterionResult.invalid_reason`) |
| `low_coverage` | a block has fewer valid samples than `min_block_coverage` |
| `insufficient_history` | less than one window of samples since the start or a reset |
| `missing` / `unreadable` | a tolerated non-finite or unreadable sample |
| `gap_reset` | missing data exceeded `max_gap_s` and cleared the history |

With `hold_when_undecided`, rows with `undecided`, `invalid_result` or
`low_coverage` keep the last decided label, and the reason says why that
window was not decided.

## Segments, fits and statistics by state

Once a record is labelled, `rocksteady` can split it into segments, fit
each segment, and report statistics by state.

```python
report = rs.batch.fit_report(
    detector, df,
    models={"level": {"steady": 0, "transient": "first_order"}, "actuator": 1},
)
print(report)          # tables by state and by segment
report.by_state        # DataFrame indexed by (signal, state)
report.by_interval     # DataFrame, one row per (signal, segment)
```

```
level   fit vs data   (models: steady poly 0, transient first_order, indeterminate first_order)
  state              n       time    bias    std    MAE   RMSE  max|e|  NRMSE       r  u(bias)  tau_int  N_eff
  all            7,200     2.00 h  +0.000  2.438  1.223  2.438  23.136  0.78%  1.0000   0.2183     58 s    125
  steady         6,457  107.6 min  +0.000  2.527  1.261  2.526  23.136  0.80%  1.0000   0.2434     60 s    108
  transient        743   12.4 min  +0.000  1.453  0.891  1.453   6.138  0.51%  1.0000   0.0533      1 s    743
```

### States

`rs.segments(labels)` (or `rs.segments_from(detector, batch.run(...))`)
splits a record into segments: contiguous runs, each in one state:

| State | From the labels |
|---|---|
| steady | a run that reached `steady`, with any `holding` just before it |
| transient | `disturbed`, `settling`, and `holding` that never confirmed |
| indeterminate | `indeterminate` outside steady segments: start of record, data gaps, windows too noisy to decide |

The "all" rows cover every sample.

Labels are causal: a sample is called steady only once the window *ending*
at it passes. `segments_from` therefore **backfills** each steady
segment over the window that certified it: the `W - 1` samples before
its first steady sample, for a window of `W` samples. It does not backfill runs that
an override forced to steady (they were not certified by data), and it
never backfills across a disturbance. A segment that holds any
override-forced label has `forced=True`; the per-segment tables carry a
`forced` column and print its state as `steady (forced)`. During slow approaches, labels can
flicker between `settling` and `indeterminate`. `min_samples=` merges runs
shorter than that into their neighbours.

### Fit models

The model for each signal is chosen per state. An integer is a polynomial
order. Any other model is named, or given as `{"model": ..., **options}`:

| Model | Form | Parameters |
|---|---|---|
| `poly` (order *p*) | polynomial on scaled time | `level`, `slope`, `curvature`, … at the segment centre |
| `first_order` | `y∞ + (y0 − y∞)·exp(−(t − t0)/τ)` | `y0`, `y_inf`, `tau`, `lambda`; `y0=0` gives `A(1 − exp(−λt))` |
| `exp` | `c + A·exp((t − t0)/T)` | `c`, `A`, `T` (growth T > 0, decay T < 0; a reactor period) |

Missing `indeterminate` entries fall back to `transient`. The exponential
models are fitted by variable projection, with numpy only. Their time
constants come with uncertainties corrected for autocorrelation.

`tau_unidentifiable` flags a time constant the segment cannot determine:
- the segment is shorter than τ/2, so the curve looks like a line;
- τ is under five samples, so it looks like a step;
- or u(τ) > τ/2.

### The fit is the detrend

Noise statistics (`std`, `tau_int`, `N_eff`) come from the residuals about
each segment's own model, so **the model decides what counts as noise**:
- **Too simple:** a constant fitted to a drifting segment leaves the drift in the "noise". The tell-tale is a `tau_int` much longer than the sensor's time constant.
- **Too flexible:** an overly flexible model absorbs slow noise and makes the uncertainty look better than it is.

Choose models from the physics.

### Three kinds of statistics

| What | Error | Columns |
|---|---|---|
| One signal, per segment (`metrics.signal_stats`) | residual about the fit | mean, std, `tau_int`, `N_eff`, `u(mean)`, fitted parameters ± u |
| Live vs fit, per state (`report.fit_report`) | `e = fit − data` | n, time, bias, std, MAE, RMSE, max\|e\|, NRMSE, r, `u(bias)`, `tau_int`, `N_eff` |
| Live vs live, per state (`report.compare_report`) | `e = a − b` | the same |

The column conventions:
- **Bias:** `mean(e)`.
- **NRMSE:** normalised by `|mean(b)|` by default; `nrmse="range"`, `"std"` or `"none"` are also available. With long shutdowns the mean is small, so `"range"` may read better.
- **r:** Pearson's correlation, `—` when either series is constant (an order-0 fit over one plateau).

### Uncertainty and autocorrelation

Neighbouring samples are correlated, so `std/√n` understates the uncertainty of a mean.

- **The correction:** `rocksteady` estimates the integrated autocorrelation time `tau_int` and uses the effective sample count `N_eff = duration / tau_int`. It reports `u(mean) = std/√N_eff` and `u(bias) = std_e/√N_eff`. The spread itself is not corrected.
- **Pooling:** autocorrelation is estimated within contiguous pieces only: never across segment boundaries, non-finite samples or omitted timestamps.
- **Comparisons:** errors in comparisons are de-meaned but not detrended. A slowly wandering offset is real uncertainty in the bias.
- **Low N_eff:** `*` marks `N_eff < 10`, where a normal coverage factor is not appropriate.

In Monte Carlo checks with AR(1) noise, `u(mean)` and `u(bias)` were within
about 1 % of the true uncertainty for φ up to 0.9. Coverage of `τ ± 2u(τ)`
was 93–97 % (nominal 95.4 %).

Fit parameter uncertainties (`uncertainty="ar1"`, the default) treat
consecutive valid samples as one sample apart. A fit whose segment has
missing samples carries the flag `time_gaps`. With up to 60 % of samples
missing, the level uncertainty stayed within about +10 % of the truth in
Monte Carlo checks. `uncertainty="ar1_time"` models the actual sample
positions instead; it was no better calibrated, so it is opt-in.

`stats.mean_with_uncertainty(x, dt, u_type_b=None)` remains for a single
steady segment. It detrends with a line and combines an optional Type B
uncertainty in quadrature.

### Provenance, evaluation and export

`report.provenance` records how a report was made:
- the rocksteady version;
- the time origin and sample spacing;
- every signal's model spec per state, with defaults filled in (`fit.normalize_model`), and its parameter order;
- the uncertainty methods and the NRMSE normalization;
- from `batch.fit_report`, the detector's effective timing, validity policy and components, and the segmentation settings.

The tables carry it in `.attrs["rocksteady"]`. `report.to_json(path)` writes
it with both tables and every segment's fit (`FitResult.to_dict()`).

Fitted parameters are stated at a reference time `t_ref`: the segment
centre for `poly`, and the first valid sample for the exponential models. The
`t_ref_s` column gives it. `FitResult.evaluation` is a numerically stable
form; for `exp` growth it holds the amplitude at the segment end, which
cannot underflow. `FitResult.predict(t)`, or
`fit.evaluate(model, evaluation, t)` on an exported record, reproduces the
fitted curve at any time on the same clock.

## Best-fit segments under constraints

`rs.optimal_segments` answers a different question from the detector: *what
is the best piecewise description of this record, given a budget of
segments?* It chooses breakpoints and a state for each segment, fits each
segment with the model for its state, and minimises the total squared error
(so the RMS of fit − data), subject to your constraints:

```python
res = rs.optimal_segments(
    df.index, df["power"],                       # a DatetimeIndex, or seconds
    models={"steady": 0, "transient": "exp"},
    per="1D", max_counts={"steady": 3, "transient": 2},
    min_duration_s={"steady": 600},
    max_duration_s={"transient": 7200},          # the main lever on run time
    method="refine",
    candidates=rs.segments_from(detector, out),  # detector boundaries as a starting point
    refine_s=detector.timing.evaluated_window_s,
)
print(res)                                       # RMS, counts, candidates
res.counts                                       # segments per period and state
report = rs.report.fit_report(values, res.segments, {"power": {"steady": 0, "transient": "exp"}})
rs.compare_segments(res.segments, rs.segments_from(detector, out))  # agreement with the detector
```

These are **best-fit segments, not certified steady states**: the optimiser
uses the segments it is allowed, whether or not the data would pass the
detector's criteria.

| Constraint | Meaning |
|---|---|
| `per`, `max_counts` | at most N segments of a state per period. A segment counts once against every period it touches: one segment from 16:00 to 08:00 the next day counts against both days. Periods are half-open and follow a `DatetimeIndex`'s wall clock. |
| `transitions` | which state may follow which; default any. This constrains the sequence of states only. Segments are fitted independently, so a level change may sit at any boundary, including transient → steady. |
| `min_duration_s`, `max_duration_s` | per state |
| `penalty` | added per segment. The default, `"bic"`, makes the counts maxima rather than quotas; `0` uses every allowed slot. |

| `method` | Candidate breakpoints | Result |
|---|---|---|
| `"subset"` | `candidates` (e.g. detector boundaries) | the best choice among them |
| `"refine"` | `subset`, then each boundary moved within `refine_s` | a local optimum, never worse than `subset` |
| `"exact"` | every sample, or every `grid_s`, plus `candidates` | the global optimum on that grid |

Optimal breakpoints for fewer segments are generally not a subset of those
for more. For example, fit `y = t` with constants: the best 2-segment split
is at 0.5, the best 3-segment split at 1/3 and 2/3. So `subset` is exact
only when the candidates contain the optimum.

On a synthetic week at 1 Hz (`python tools/benchmark.py optimal`):
- `subset` takes about 1 s;
- `refine` takes about 25 s with exponential transients, almost all of it in
  the exponential fits, and under 1 s with linear ones;
- `exact` with polynomial models takes under 1 s on a 5-minute grid and about
  14 s on a 1-minute grid.

Constant and linear segment costs come from running sums. Other models are
fitted exactly as `fit_report` fits them.

## Extending rocksteady

Every component is registered by name, so your own components work in Python
code and in specs alike:

```python
import numpy as np
import rocksteady as rs
from rocksteady import registry, Criterion, CriterionResult

@registry.criteria.register("range")
class RangeCriterion(Criterion):
    def __init__(self, signal, limit, noise, k=3.0, name=None):
        self.signal, self.limit, self.noise, self.k = signal, limit, noise, k
        self.name = name or f"{signal}_range"

    def evaluate(self, t_blocks, y_blocks):
        return CriterionResult(self.name, float(np.ptp(y_blocks)), self.noise, self.limit, self.k)

rs.criterion("range", signal="T1", limit=0.2, noise=0.01)
# or, in a spec: {"type": "range", "signal": "T1", "limit": 0.2, "noise": 0.01}
```

| Registry | Base / protocol |
|---|---|
| `registry.criteria` | `Criterion.evaluate(t_blocks, y_blocks) -> CriterionResult` |
| `registry.disturbances` | `Disturbance.update(t, sample, values) -> bool` (and `reset`) |
| `registry.overrides` | `Override.apply(ctx) -> Label or None` (and `reset`) |
| `registry.signal_builders` | callable returning a source (a column name or `sample -> float`) |
| `registry.slope_methods` | `(t, y, **options) -> (b, u_b, k)` |
| `registry.fit_models` | `(t, y, **options) -> FitResult` |

Slope methods and fit models must accept `(t, y)` positionally; the
protocols are `registry.SlopeMethod` and `registry.FitModel`. Give a
provider a `validate_options(**options)` attribute, and `DriftCriterion` (for
slope methods) and `fit.validate_models` (for fit models) reject bad option
names and values when the detector or report is set up, not partway
through a long run.

Spec entries may also be built objects. That is how a criterion with a
callable noise model goes into a spec.

**Stateful components belong to one detector.** Disturbances and overrides
keep state (a step detector's history, a latch), and the detector updates
and resets it. Do not share an instance between detectors that are used
at the same time. `build_detector(spec)` with plain-data entries builds
fresh instances on every call, and rocksteady never copies components
behind your back.

## Performance

`batch.run` loops over samples in Python. With 30 s blocks and 1 Hz data,
expect a few tens of microseconds per criterion per sample (about 4 s for
four hours of six drift criteria). The AR(1) drift test costs O(n) in the
number of blocks, so long windows are affordable. `python tools/benchmark.py`
measures the scaling.

## References

- Bellman, R. (1961). On the approximation of curves by line segments using dynamic programming. *Communications of the ACM* 4(6), 284. (Optimal segmentation by dynamic programming.)
- Cao, S. and Rhinehart, R. R. (1995). An efficient method for on-line identification of steady state. *Journal of Process Control* 5(6), 363–374.
- Cao, S. and Rhinehart, R. R. (1997). Critical values for a steady-state identifier. *Journal of Process Control* 7(2), 149–152. (The filtered R-statistic, a planned criterion.)
- Golub, G. H. and Pereyra, V. (1973). The differentiation of pseudo-inverses and nonlinear least squares problems whose variables separate. *SIAM Journal on Numerical Analysis* 10(2), 413–432. (Variable projection, used by the exponential fit models.)
- Golub, G. and Pereyra, V. (2003). Separable nonlinear least squares: the variable projection method and its applications. *Inverse Problems* 19(2), R1–R26.
- Jackson, B. et al. (2005). An algorithm for optimal partitioning of data on an interval. *IEEE Signal Processing Letters* 12(2), 105–108. (Optimal partitioning, the basis of `optimal_segments`.)
- JCGM 100:2008. *Evaluation of measurement data — Guide to the expression of uncertainty in measurement* (GUM). (Type A and Type B standard uncertainties.)
- Marriott, F. H. C. and Pope, J. A. (1954). Bias in the estimation of autocorrelations. *Biometrika* 41, 390–402. (First-order bias of the lag-1 autocorrelation. `"ar1"` uses the analogous correction for a series detrended by a line, and is validated by Monte Carlo.)
- Neumaier, A. (1974). Rundungsfehleranalyse einiger Verfahren zur Summation endlicher Summen. *ZAMM* 54(1), 39–51. (Compensated summation, used for the segment-cost range sums.)
- Newey, W. K. and West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica* 55(3), 703–708. (The `"nw"` method.)
- Schuirmann, D. J. (1987). A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability. *Journal of Pharmacokinetics and Biopharmaceutics* 15(6), 657–680. (Equivalence testing.)
- Schwarz, G. (1978). Estimating the dimension of a model. *Annals of Statistics* 6(2), 461–464. (The BIC, the default segment penalty.)
- Sokal, A. (1997). Monte Carlo methods in statistical mechanics: foundations and new algorithms. In *Functional Integration: Basics and Applications*, NATO ASI Series B 361, 131–192. (The integrated autocorrelation time and its windowed estimation.)
- Truong, C., Oudre, L. and Vayatis, N. (2020). Selective review of offline change point detection methods. *Signal Processing* 167, 107299. (Context: `ruptures`.)
- Wellek, S. (2010). *Testing Statistical Hypotheses of Equivalence and Noninferiority*, 2nd ed. Chapman & Hall/CRC. (The three-way test as an equivalence test.)
- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica* 48(4), 817–838. (Sandwich covariance.)
- Zhang, N. F. (2006). Calculation of the uncertainty of the mean of autocorrelated measurements. *Metrologia* 43, S276–S281. (Uncertainty of a mean with autocorrelation, and combining Type A and Type B. `tau_int` uses a different autocorrelation cutoff; see its docstring.)

The mathematics of each method, as implemented, is on the documentation's
[Methods](https://rocksteady.readthedocs.io/en/latest/guide/methods.html) page.

## Development

```
pip install -e ".[dev]"
tools/check.sh                      # what CI runs, fast: tests without the slow ones, mypy, strict docs build
tools/check.sh tests/test_optimal.py -k infeasible  # one selection of those tests, no mypy or docs build
python -m pytest                    # everything
python -m pytest -m "not slow"      # skip the Monte Carlo calibration
python tests/test_regression.py     # regenerate golden snapshots after an intended label change
python tests/test_optimal_snapshot.py  # regenerate the best-fit segments snapshot after an intended change
```

See [CHANGELOG.md](https://github.com/UT-Computational-NE/rocksteady/blob/main/CHANGELOG.md) for changes between versions.

## License

BSD 3-Clause. See [LICENSE](https://github.com/UT-Computational-NE/rocksteady/blob/main/LICENSE).

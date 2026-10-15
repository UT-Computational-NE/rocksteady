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

## Install

```
pip install "rocksteady @ git+https://github.com/<org>/rocksteady.git@v0.2.0"
```

Python 3.10 or later, with numpy and pandas. pandas is used only by
`rocksteady.batch`; the streaming detector needs numpy alone.

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

Runnable examples with synthetic data are in [`examples/`](examples):

- [`thermal_drift.py`](examples/thermal_drift.py): channels relaxing to a setpoint, with an event log, settling time and hold time.
- [`actuator_steps.py`](examples/actuator_steps.py): a level held by a stepping actuator, with a level-dependent noise model and a shutdown rule.

## Concepts

A detector is built from five kinds of parts. The state machine that
combines them is shared.

| Part | What it is | Built in |
|---|---|---|
| **Signals** | Named scalars computed from each raw sample: a column, or a derived quantity | column, `mean`, `spread`, `difference` |
| **Criteria** | Tests on one signal over the trailing window of block means. Each returns a statistic, its uncertainty, and a limit | `DriftCriterion` (no resolvable slope), `DeviationCriterion` (stays within a tolerance of a constant or a line), `BoundCriterion` (mean inside a band) |
| **Disturbances** | External action, found in the signals or read from a log; resets the clocks | `StepChange` (a discrete move in a signal), `EventLog` (logged events with durations) |
| **Timing** | Sample period, block length, window, settling time after a disturbance, hold time, gap tolerance, and what an undecided window means | `Timing` |
| **Overrides** | Domain rules that force a label | `FloorLatch` (below a floor the system is steady by definition, for example when shut down) |

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

## Gaps and timestamps

Times passed to `update` must increase. By default (`Timing.max_gap_s = 0`),
the detector clears its history on any of these:

- a non-finite sample;
- an unreadable sample;
- a step in time of more than 1.5 sample periods.

With `max_gap_s > 0`, shorter gaps are kept as missing data. Block means
then ignore the missing samples, and a window is evaluated only if every
block keeps at least `min_block_coverage` of its samples.
`rocksteady.batch.run` takes sample times from a `DatetimeIndex` when there is one.

## Reporting a steady value

`rocksteady.stats.mean_with_uncertainty(x, dt, u_type_b=None)` gives the mean over an
accepted interval and its uncertainty. The Type A part uses an effective
sample count `N_eff = duration / tau_int`, so correlated noise does not
understate it. A Type B uncertainty, if given, is combined in quadrature.
`low_effective_sample` flags `N_eff < 10`.

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

Spec entries may also be built objects. That is how a criterion with a
callable noise model goes into a spec.

## Performance

`batch.run` loops over samples in Python. With 30 s blocks and 1 Hz data,
expect a few tens of microseconds per criterion per sample (about 4 s for
four hours of six drift criteria).

## References

- Newey, W. K. and West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica* 55(3), 703–708. (The `"nw"` method.)
- Marriott, F. H. C. and Pope, J. A. (1954). Bias in the estimation of autocorrelations. *Biometrika* 41, 390–402. (First-order bias of the lag-1 autocorrelation. `"ar1"` uses the analogous correction for a series detrended by a line, and is validated by the Monte Carlo above.)
- Zhang, N. F. (2006). Calculation of the uncertainty of the mean of autocorrelated measurements. *Metrologia* 43, S276–S281. (Uncertainty of a mean with autocorrelation, and combining Type A and Type B. `tau_int` uses a different autocorrelation cutoff; see its docstring.)
- Cao, S. and Rhinehart, R. R. (1995). An efficient method for on-line identification of steady state. *Journal of Process Control* 5(6), 363–374.
- Cao, S. and Rhinehart, R. R. (1997). Critical values for a steady-state identifier. *Journal of Process Control* 7(2), 149–152. (The filtered R-statistic, a planned criterion.)

## Development

```
pip install -e ".[dev]"
python -m pytest                    # everything
python -m pytest -m "not slow"      # skip the Monte Carlo calibration
python tests/test_regression.py     # regenerate golden snapshots after an intended label change
```

See [CHANGELOG.md](CHANGELOG.md) for changes between versions.

## License

MIT. See [LICENSE](LICENSE).

# Changelog

## Unreleased

### Added

- `rocksteady.states`: `State` (steady, transient, indeterminate),
  `Segment`, `segments(labels, backfill_samples=, min_samples=)` and
  `segments_from(detector, labels)`, which split a labelled record into
  intervals by state. Steady segments are the accepted intervals,
  optionally backfilled over their certifying window (#1).
- `batch.signal_values(detector, df)`: the detector's signal values for
  every row (#1).
- `rocksteady.fit`: fit models for intervals, registered in
  `registry.fit_models` (#2).
  - The `"poly"` model fits a polynomial of any order on scaled time.
    Its parameters are the derivatives at the interval centre: `level`,
    `slope`, `curvature`, ...
  - Parameter covariances are corrected for autocorrelation with an
    AR(1) sandwich, computed in O(n).
  - `piecewise_fit` fits each segment with the model for its state; the
    model spec is per signal and optionally per state.
- Exponential fit models, fitted by variable projection with numpy only (#3):
  - `"first_order"` (`y_inf + (y0 - y_inf) exp(-t/tau)`; option `y0=0`
    gives `A(1 - exp(-lambda t))`);
  - `"exp"` (`c + A exp(t/T)`, with T a reactor period).

  Time constants come with uncertainties corrected for autocorrelation,
  and the flags `tau_unidentifiable` and `tau_at_bound`.
- `rocksteady.metrics`, statistics with uncertainties corrected for
  autocorrelation (#4):
  - `signal_stats` (per interval, about its fit: mean, std, `tau_int`,
    `N_eff`, `u_mean`);
  - `error_stats` (`a - b`: n, time, bias, std, MAE, RMSE, max|e|, NRMSE,
    r, `tau_int`, `N_eff`, `u_bias`);
  - `stats_by_state` and `stats_by_segment`.
- `stats.acf_tau`: integrated autocorrelation time pooled over pieces,
  never across their boundaries; FFT-based for long pieces (#4).
  `pair_weighted=True` weights each lag by its share of within-piece pairs.
  This gives the time constant for the uncertainty of a mean pooled over
  separate pieces, and the metrics use it for `N_eff`.

### Changed

- `stats.tau_int` delegates to `stats.acf_tau`. Results are unchanged (to
  rounding), and long series are much faster.
- `batch.intervals` and `batch.accepted_intervals` accept any sequence of
  labels (strings or `Label`), not only a pandas Series, and return plain
  `int` indices.

## 0.2.0

### Packaging

- BSD 3-Clause license; `src/` layout; `py.typed`; version single-sourced from
  `rocksteady.__version__`; Python 3.10 or later.
- The application adapters that 0.1 shipped alongside the library are
  removed from the repository; they now live with the projects that use
  them.

### Breaking changes

- `Test` is renamed `Outcome`, and `CriterionResult.test` is renamed
  `CriterionResult.outcome`. The old name made pytest try to collect the
  enum as a test class in any test module that imported it.
- `Detector.update` raises `ValueError` if `t` does not increase.
- A step in `t` of more than 1.5 sample periods is now treated as a gap.
  With the default `Timing.max_gap_s = 0` it resets the detector, as a
  non-finite sample already did. Previously timestamps were ignored.
- `DriftCriterion` now defaults to the calibrated `"ar1"` slope
  uncertainty with a method-supplied coverage factor (`k=None`). The 0.1
  behaviour is `DriftCriterion(..., method="nw", k=2.0)`. Its arguments
  after `limit_per_s` are keyword-only, and `max_lag_frac` moved to
  `method_options={"max_lag_frac": ...}`.
  Why: in Monte Carlo checks, the 0.1 Newey-West interval falsely
  certified a drift limit 10.6 % of the time at 40 blocks with AR(1)
  phi = 0.5 (nominal 2.3 %), and 17.7 % at phi = 0.8. `"ar1"` stays at
  or under 4.8 % from 40 blocks up. It is more conservative on strongly
  autocorrelated windows.
- `rocksteady.core.linear_residuals` is removed (use
  `rocksteady.stats.linear_fit(t, y)[2]`), so the core depends on numpy
  only.

### Label changes

- At the start of a record (no disturbance since `reset`), the settling
  wait `settle_s` after the window first fills is labelled
  `indeterminate` instead of `settling`, because nothing was disturbed.
  The wait itself is unchanged. Golden snapshot `drift`: samples 299-358.

### Added

- Provider registries (`rocksteady.registry`) and factories
  (`rs.criterion`, `rs.disturbance`, `rs.override`, `rs.build_detector`).
  Every built-in component is registered by name, and user components can
  be registered the same way. `rs.build_detector(spec)` assembles a
  detector from a plain-data (JSON-compatible) spec whose entries may also
  be built objects.
- Slope-uncertainty methods are pluggable (`registry.slope_methods`):
  `"ar1"` (AR(1) error model, bias-corrected phi, Student-t factor on the
  effective degrees of freedom) and `"nw"` (0.1). `tools/coverage_mc.py`
  calibrates every registered method.
- `stats.t_quantile`: exact Student-t quantiles (integer degrees of
  freedom), without scipy.
- `stats.mean_with_uncertainty(..., u_type_b=...)` returns `u_combined`,
  the Type A and Type B parts combined in quadrature.
- The streaming detector no longer needs pandas at import time; only
  `batch.run` uses it.
- `rocksteady.signals`: `mean`, `spread`, `difference` source builders.
- `FloorLatch` override: below a floor, after a downward crossing, the
  label is forced (steady by default) until the signal recovers.
- `Timing.max_gap_s` and `Timing.min_block_coverage`: tolerate short runs
  of missing data, using NaN-aware block means.
- `Timing.undecided_counts_toward_hold` (default True, the previous
  behaviour).
- `Criterion.min_blocks`; the detector rejects windows too short for its
  criteria.
- `batch.run` uses a `DatetimeIndex` for sample times when `t` is not given.

### Documentation

- `stats.tau_int` and `mean_with_uncertainty` document their estimator
  and why it differs from Zhang (2006).
- `DeviationCriterion` documents that its interval is a calibrated
  heuristic, not a confidence interval.

### Fixed

- `CriterionResult.ratio` is infinite when the statistic or uncertainty is
  not finite, so `Verdict.eta` and `Verdict.limiting` no longer depend on
  the order of the criteria.

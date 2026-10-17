# Changelog

## Unreleased

### Added

- `rs.optimal_segments`: best-fit segments under limits (#45).
  - **Objective:** breakpoints and per-segment states that minimise the total
    squared error of each segment's fit, the model chosen per state, plus a
    per-segment penalty (default `"bic"`).
  - **Limits:** at most N segments of a state per period (`per`,
    `max_counts`); a segment counts against every period it touches, and
    periods follow a `DatetimeIndex`'s wall clock. Also allowed
    `transitions` (default: any) and per-state `min_duration_s` /
    `max_duration_s`.
  - **Methods:** `subset` (among given candidates, e.g. the detector's
    boundaries), `refine` (local search around them) and `exact` (every
    sample, or a grid). All three share one dynamic program; `exact`
    matches brute force.
  - **Speed:** constant and linear costs come from running sums. Other
    models are fitted with `fit.fit_model`, with a bound on known feasible
    cost to prune.
  - **Output:** `OptimalSegmentation` with `segments` (for `fit_report`),
    RMS, SSE and per-period counts.
  - **Helpers:** `rs.compare_segments` (sample agreement and confusion
    against, e.g., the detector) and `optimal.count_by_period`.
  - `tools/benchmark.py optimal`: a week at 1 Hz.

## 0.4.0

Fixes from the 0.3.0 review (#14-#21), validation, verdict reasons and the
missing-sample policy, run-output time metadata, performance, provenance,
provider contracts, and invariant tests and CI (#22-#28).

Upgrading from 0.3.0. Behaviour changes to check:
- A record sampled faster than `sample_period_s` (a step under half a period)
  now raises instead of being counted as full periods (#22).
- Tolerated NaN rows no longer split a steady interval; segmentation bridges
  them (#24).
- `batch.run` has new columns: `reason`, `valid` and `window_start_s`
  (#22, #24).
- Configurations, and criterion results, that were silently accepted may now
  be rejected or undecided (#23). So may duplicate signal or criterion names
  (#14).

### Fixed

- Duplicate signal or criterion names are rejected when a `Detector` is
  built. With duplicate criterion names, a later passing result used to
  overwrite an earlier failing one, so acceptance depended on criterion order.
  Every criterion's outcome is now counted independently. `batch.run` and
  `batch.signal_values` reject DataFrames with duplicate column names (#14).
- Drift slopes use the effective block duration (`block_s` rounded to whole
  samples) for the block times. With `block_s` not a whole number of
  samples, the slope was scaled by `block_s / effective_block_s`, and a drift
  above the limit could be certified. `block_s` shorter than one sample is
  rejected. `Timing` exposes `effective_block_s`, `evaluated_samples`,
  `evaluated_window_s` and `effective`. `segments_from` backfills over the
  evaluated span (#15).
- `signals.spread` is NaN when any channel is not finite. Python's `max` and
  `min` handled NaN by position, so a missing channel could be ignored and a
  record certified on partial data, depending on channel order (#16).
- `segments_from` backfills by time when the run output has `t_s`, so a
  steady segment covers exactly its certifying window even when tolerated
  timestamp gaps left samples out of the record. Counting stored rows
  could reach back past the window into rejected data. `segments` gains
  `t=` and `backfill_s=` (#17).
- `FloorLatch`: a disturbance beginning releases the latch even on the
  sample where the signal crosses the floor. Previously the crossing
  latched first, and the label was forced steady straight after the
  disturbance, skipping the settling time (#18).
- `EventLog` flags an instantaneous event at the first sample's time, and an
  event already under way at the first sample. The first sample used to
  expire such events before checking them. Events that ended before the
  record began are still not replayed (#19).
- `batch.fit_report` uses the `t_s` saved in a `batch.run` output passed as
  `labels`. It used to re-infer times from the index, so fits and interval
  times could differ from the run that produced the labels. It now rejects
  labels whose length or index do not match `df`, and an explicit `t` that
  conflicts with saved `t_s` (#20).
- `batch.run` builds its output with an explicit column schema. An empty
  record used to return only `t_s`, which broke `segments_from` and
  `batch.fit_report`. With `with_results=True` there is one
  `ratio_<criterion>` column per criterion. `stats_by_segment`, and the
  interval tables of `fit_report` and `compare_report`, keep their columns
  when there are no segments (#21).

### Changed

- Configurations are validated when they are built, whether by constructor
  or by plain-data spec (the same checks run either way); errors name the
  component and option (#23):
  - finite timing values;
  - non-empty names, and signal references in disturbances and overrides;
  - finite, ordered bounds and positive limits;
  - positive-integer averaging lengths;
  - non-negative event durations;
  - `FloorLatch` floor > 0 and release factor >= 1.

  Some configurations accepted before are now rejected.
- A criterion result with a negative or non-finite uncertainty or coverage
  factor, or a non-positive limit, is undecided. It used to pass.
  `CriterionResult.invalid_reason` says why (#23).
- Fit search bounds (`tau_min`/`tau_max`, `T_min`/`T_max`) must be finite,
  positive and ordered. `validate_models` checks option names and values
  before any data is processed, using `fit.check_model_options` and each
  provider's `validate_options` (#23).
- `states.check_segments` rejects segments outside the record or overlapping
  each other, and allows partial coverage. `piecewise_fit` and the
  statistics use it (#23).
- The missing-sample hold policy is specified: a tolerated missing sample
  is `indeterminate` (reason `missing` or `unreadable`) whatever the
  undecided options, and the hold clock pauses over it, as over omitted
  timestamps. `segments`/`segments_from` bridge such samples inside a
  steady run (`segments(..., bridge=)`, from the `reason` column), so NaN
  rows and omitted timestamps give the same steady intervals. A steady run
  is no longer split by tolerated NaN rows (#24).
- Sampling contract: `update` rejects a step shorter than half a sample
  period (a record sampled faster than `sample_period_s`), which was
  previously counted as a full period. `Timing.grid_tolerance` (default
  0.5) can be lowered to also reject steps off the sampling grid (#22).

### Added

- Verdict reasons: `Verdict.reason` (`rs.Reason`) says why a sample has its
  label (certified, forced, disturbance, settling time, failed, undecided,
  invalid result, low coverage, insufficient history, missing, unreadable,
  gap reset). `batch.run` returns it as a `reason` column (#24).
- Forced-versus-certified provenance downstream: `Segment.forced` is True
  when an override forced a label in the segment, and the interval tables
  of `fit_report`, `compare_report` and `stats_by_segment` carry a `forced`
  column (#24).
- `batch.run` adds `valid` (the sample was readable and finite) and
  `window_start_s` (start of the window examined on each row: a passing
  row's certifying window), from `Verdict.valid` and
  `Verdict.window_start_s`. Output rows map one-to-one to input rows, with
  the same index (#22).
- `error_stats`, `signal_stats`, `stats_by_state` and `stats_by_segment`
  take sample times `t=`, and `fit_report`/`compare_report` pass them.
  Autocorrelation is no longer estimated across omitted timestamps, where
  adjacent stored rows are not adjacent observations
  (`metrics.split_at_time_gaps`) (#22).
- `tools/benchmark.py`: scaling of `slope_ar1`, `Detector.update`
  (blocks, channels, window copy), fragmented `acf_tau` and `EventLog` (#25).

- Provenance: `Report.provenance` records the rocksteady version, time origin
  and sample period, normalized model specs (`fit.normalize_model`) and
  parameter order per signal and state, uncertainty methods and NRMSE
  normalization. `batch.fit_report` adds the detector's effective timing,
  validity policy and components (`batch.detector_provenance`) and the
  segmentation settings. The tables carry it in `.attrs["rocksteady"]`, and
  `Report.to_dict()`/`to_json()` export it with the tables and fits (#26).
- Stable evaluation: `FitResult.t_ref` (when `params` are stated),
  `FitResult.evaluation` (a stable, JSON-ready form: for `exp` growth the
  amplitude at the interval end, which cannot underflow), `FitResult.predict(t)`,
  `FitResult.to_dict()` and `fit.evaluate(model, evaluation, t)`. Interval
  tables gain `t_ref_s` (#26).
- Fits flag `time_gaps` when the valid samples are not consecutive. This
  makes the AR(1) assumption visible: consecutive valid samples are one
  sample apart. `uncertainty="ar1_time"` (opt-in) models the actual sample
  positions; `_ar1.ar1_apply(..., steps=)` and `_ar1.gap_steps` support it (#26).

- Provider protocols `registry.SlopeMethod`, `FitModel` and `SignalBuilder`.
  Slope methods and fit models must accept `(t, y)` positionally, which is
  checked when they are registered. `registry.check_options` binds options to
  a provider's signature and calls its optional `validate_options`. Fit
  models use it (`fit.check_model_options`), and so does `DriftCriterion`
  for `method_options`: a misspelled or out-of-range slope option fails
  when the detector is built, naming the criterion, method and option. The
  built-in `ar1` and `nw` methods validate their options (#27).

### Tests and CI

- `tests/test_invariants.py` adds cross-cutting invariants. The focused
  regressions stay with their fixes (#14-#21). The invariants cover:
  - criterion renaming and reordering;
  - derived-signal missingness under channel permutation;
  - units, which scale slopes and levels but not time constants or labels;
  - omitted timestamps vs missing rows;
  - backfill confined to the certifying window and never across a
    disturbance or reset;
  - empty vs non-empty schemas.

  The deterministic snapshots and the Monte Carlo calibration are kept (#28).
- CI adds a `min-deps` job: Python 3.10 with numpy 1.22 and pandas 1.5, the
  minimums in `pyproject.toml`. The `build` job runs the extracted sdist's
  tests against the installed wheel, outside the source tree, with no
  editable install or `PYTHONPATH` (#28).

### Documentation

- Dependencies: pandas stays a required dependency. The streaming detector
  needs numpy only and does not import pandas, which is tested. `batch`,
  `metrics` and `report` use pandas (the docs said only `batch`). Making
  pandas an optional extra was considered and deferred; it would break
  `pip install rocksteady` for every batch user (#27).
- Stateful components (disturbances, overrides) belong to one detector.
  `build_detector(spec)` builds fresh ones on every call, and rocksteady
  never copies components (#27).
- The `factory` example uses `rs.factory.signal(...)`; `rs.signal` was never
  exported (#27).

### Performance

- `stats.slope_ar1` no longer forms dense n x n lag and correlation
  matrices, and the unbounded process-wide lag-matrix cache is gone. It
  uses the O(n) AR(1) primitive `ar1_apply`, moved with a new `ar1_quad` to
  the dependency-free module `rocksteady._ar1` (`rocksteady.fit.ar1_apply`
  still works). 10,000 blocks: 624 ms to 1.8 ms per call; 90 blocks: 74 to
  33 us. Uncertainties agree with the dense algorithm to 1e-11 (relative);
  slopes and coverage factors are identical (#25).
- `ar1_apply` runs its recursions on Python floats: same results bit for
  bit, several times faster (fit covariances of long intervals) (#25).
- `stats.acf_tau` accumulates only the lags each piece has. One day-long
  piece plus 10,000 short ones: 461 ms to 52 ms, identical results (#25).

## 0.3.0

Interval fits and statistics by state (milestone v0.3.0, #1-#6).

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
- `rocksteady.report` (#5):
  - `fit_report` (each signal against its piecewise fit, by state and by
    interval, with fitted parameters);
  - `compare_report` (`a` against `b`, by state and by interval);
  - `Report`, with `by_state` and `by_interval` DataFrames, `fits`, and
    plain-text tables.
- `batch.fit_report(detector, df, models)`: run, segment and report in one
  call (#5).
- Examples print fit and comparison reports; README section "Intervals,
  fits and statistics by state" (#6).

### Notes

- `segments_from` takes the `batch.run` output and backfills exactly the
  certifying window (`window_samples - 1` samples). Override-forced
  steady runs are not backfilled, and backfill never crosses a
  disturbance.
- Fit parameter covariances do not depend on parameter units. Parameters
  the data cannot determine get infinite variance and the flag
  `rank_deficient`.

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

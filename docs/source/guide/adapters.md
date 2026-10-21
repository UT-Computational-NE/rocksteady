# Writing an adapter

rocksteady knows nothing about any facility. An **adapter** is the
application package that connects a facility's data to it:
- which raw columns become signals;
- which criteria apply, with their calibrated limits and noise models;
- which events and actions disturb the system;
- which overrides force a label (domain knowledge, such as a floor below
  which the system is steady);
- how results are reported.

Keep everything facility-specific in the adapter: channel names, limits,
noise models, time zones, model choices. rocksteady supplies only the
generic machinery.

The adapters in use follow the same four-part shape:

1. **Configuration**: the calibrated values, in one frozen dataclass.
2. **The detector**: a `build_detector()` that assembles a spec.
3. **Streaming**: a thin wrapper around {py:meth}`Detector.update <rocksteady.core.Detector.update>` for live data.
4. **Batch**: labels, segments and reports for a stored record.

The two examples on this page are complete, runnable files in
`docs/source/examples/`, and the test suite runs them.

## The calls an adapter makes

```{eval-rst}
.. currentmodule:: rocksteady

.. autosummary::

   Signal
   factory.signal
   Timing
   DeviationCriterion
   DriftCriterion
   BoundCriterion
   AsymptoteCriterion
   StepChange
   EventLog
   FloorLatch
   build_detector
   Detector
   Verdict
   suggest_timing
   batch.run
   batch.accepted_intervals
   batch.signal_values
   segments_from
   batch.fit_report
   report.fit_report
   report.compare_report
   optimal_segments
```

## A minimal adapter

A heated loop with three thermocouples and an operator log. Every name and
number here is illustrative.

### 1. Configuration

Put every calibrated value in one place, so a calibration change is a
one-line diff and the detector can be rebuilt from it:

```{literalinclude} ../examples/adapter_basic.py
:language: python
:start-after: "# [config]"
:end-before: "# [config]"
```

### 2. The detector, from a spec

A spec is plain data: JSON, YAML or TOML compatible. Every entry names a
registered component and its arguments. An entry can also be a built object
(a `Signal`, a criterion, a `Timing`); that is how callables such as a
level-dependent noise model get into a spec.

```{literalinclude} ../examples/adapter_basic.py
:language: python
:start-after: "# [spec]"
:end-before: "# [spec]"
```

```{eval-rst}
.. autofunction:: rocksteady.build_detector
   :no-index:

.. autoclass:: rocksteady.Timing
   :no-index:
   :no-members:
```

**Signals** are named scalars computed from each raw sample: a column, or a
builder over several columns.

```{eval-rst}
.. autoclass:: rocksteady.Signal
   :no-index:
   :no-members:

.. autofunction:: rocksteady.factory.signal
   :no-index:

.. autofunction:: rocksteady.signals.mean
   :no-index:

.. autofunction:: rocksteady.signals.spread
   :no-index:

.. autofunction:: rocksteady.signals.difference
   :no-index:
```

**Criteria** test one signal over the window of block means; see
{doc}`methods` for their statistics.

```{eval-rst}
.. autoclass:: rocksteady.DeviationCriterion
   :no-index:
   :no-members:

.. autoclass:: rocksteady.DriftCriterion
   :no-index:
   :no-members:

.. autoclass:: rocksteady.BoundCriterion
   :no-index:
   :no-members:

.. autoclass:: rocksteady.AsymptoteCriterion
   :no-index:
   :no-members:
```

**Disturbances** reset the sequence: the detector waits out the settling
time before it evaluates again. **Overrides** may force a label.

```{eval-rst}
.. autoclass:: rocksteady.StepChange
   :no-index:
   :no-members:

.. autoclass:: rocksteady.EventLog
   :no-index:
   :no-members:

.. autoclass:: rocksteady.FloorLatch
   :no-index:
   :no-members:
```

Logged events usually come from an operator log or a control system. The
adapter converts them to `(start_s, duration_s)` pairs on the same clock as
the record's `t`. With a `DatetimeIndex`, that is seconds since the first
row.

### 3. Streaming

`Detector.update(t, sample)` takes one sample, a mapping of column names to
values, and returns a `Verdict`. Wrap it in whatever interface the
application already uses:

```{literalinclude} ../examples/adapter_basic.py
:language: python
:start-after: "# [stream]"
:end-before: "# [stream]"
```

```{eval-rst}
.. autoclass:: rocksteady.Detector
   :no-index:
   :members: update, reset

.. autoclass:: rocksteady.Verdict
   :no-index:
   :members: eta, limiting, is_steady
```

The detector owns its disturbances and overrides: it updates and resets
their state. Build one detector per stream, for example by calling
`build_detector(spec)` once per stream; do not share component instances.

### 4. Batch

For a stored record, `batch.run` gives exactly the labels the live detector
would have given. Segmentation and reports build on its output:

```{literalinclude} ../examples/adapter_basic.py
:language: python
:start-after: "# [batch]"
:end-before: "# [batch]"
```

```{eval-rst}
.. autofunction:: rocksteady.batch.run
   :no-index:

.. autofunction:: rocksteady.batch.accepted_intervals
   :no-index:

.. autofunction:: rocksteady.segments_from
   :no-index:

.. autofunction:: rocksteady.batch.fit_report
   :no-index:

.. autofunction:: rocksteady.report.compare_report
   :no-index:
```

Pass the `batch.run` output, not just its labels, to `segments_from` and
`fit_report`. It carries the sample times, the forced labels and the
reasons that segmentation needs. For best-fit segments under constraints, see
{doc}`optimal`.

The whole file:

```{literalinclude} ../examples/adapter_basic.py
:language: python
```

## Custom components

When the built-in components do not express a facility's physics, write a
component. Register it by name, and specs can use it like any built-in one.
Each kind has a small contract:

| Kind | Base or protocol | You implement | Registry |
|---|---|---|---|
| criterion | {py:class}`~rocksteady.core.Criterion` | `evaluate(t_blocks, y_blocks) -> CriterionResult`; set `name`, `signal`, optionally `min_blocks`, `detail_keys` (diagnostics in `CriterionResult.details`), `bind_timing(timing)` (called once by the detector) and `uses_block_masks` (receive `valid=`, the per-block sample validity) | `registry.criteria` |
| disturbance | {py:class}`~rocksteady.core.Disturbance` | `update(t, sample, values) -> bool` (True while active), `reset()`; set `name` | `registry.disturbances` |
| override | {py:class}`~rocksteady.core.Override` | `apply(ctx: Context) -> Label or None`, optionally `reset()`; set `name` | `registry.overrides` |
| slope method | {py:class}`~rocksteady.registry.SlopeMethod` | `(t, y, **options) -> (b, u_b, k)`; optionally `validate_options(**options)` | `registry.slope_methods` |
| fit model | {py:class}`~rocksteady.registry.FitModel` | `(t, y, **options) -> FitResult`; optionally `validate_options`, `evaluate(evaluation, t)` | `registry.fit_models` |
| signal builder | {py:class}`~rocksteady.registry.SignalBuilder` | `(**params) -> source` (a column name or `sample -> float`) | `registry.signal_builders` |

```{literalinclude} ../examples/adapter_custom.py
:language: python
:start-after: "# [criterion]"
:end-before: "# [criterion]"
```

```{literalinclude} ../examples/adapter_custom.py
:language: python
:start-after: "# [disturbance]"
:end-before: "# [disturbance]"
```

```{literalinclude} ../examples/adapter_custom.py
:language: python
:start-after: "# [override]"
:end-before: "# [override]"
```

A slope method plugs into `DriftCriterion(method=...)`. Its options are
checked against its signature, and by its `validate_options`, when the
criterion is built:

```{literalinclude} ../examples/adapter_custom.py
:language: python
:start-after: "# [slope]"
:end-before: "# [slope]"
```

All four used by name, in plain data:

```{literalinclude} ../examples/adapter_custom.py
:language: python
:start-after: "# [spec]"
:end-before: "# [spec]"
```

```{eval-rst}
.. autoclass:: rocksteady.CriterionResult
   :no-index:
   :members: ratio, outcome, invalid_reason

.. autoclass:: rocksteady.Context
   :no-index:
   :no-members:

.. automethod:: rocksteady.registry.Registry.register
   :no-index:

.. autofunction:: rocksteady.registry.check_options
   :no-index:
```

The contract:
- A criterion's uncertainty must be a standard uncertainty: non-negative and
  finite. An invalid result (`CriterionResult.invalid_reason`) is
  undecided, never a pass.
- Disturbances see every valid sample, in time order. They are reset with
  the detector and after a gap that clears the history.
- Overrides are called with the {py:class}`~rocksteady.core.Context` of every
  readable, finite sample, disturbed ones included. A disturbance still takes
  precedence over a label they force. A sample with a missing, non-finite or
  unreadable signal skips them: the detector labels it INDETERMINATE (reason
  `missing` or `unreadable`) without calling `apply`. So an override that
  tracks a raw column can miss a change made while another required signal
  is unavailable. A gap that exceeds `Timing.max_gap_s` resets overrides with
  the history.
- Names must be unique among a detector's signals and among its criteria.

## Choosing block and window lengths

`Timing` has no default block or window length, because the right values
depend on two time scales of the application's own data:

| | Should be | Why |
|---|---|---|
| block length | several times the **noise correlation time** of the raw samples | so block means are nearly independent; too short, and they stay correlated, uncertainties grow and the drift test certifies little |
| block length | short compared with the **process time scale** | so blocks still follow a transient instead of averaging it away |
| window | about **40 blocks** or more | the `ar1` drift test holds its calibration from about 40 blocks; below about 20 no slope method is reliable |
| window | no longer than the **reaction time** you accept | a steady call lags by about one window, plus `settle_s` |

`rs.suggest_timing` measures both time scales from your data and proposes
lengths, with its reasoning. You review the proposal and adopt it; it sets
nothing itself:

```python
s = rs.suggest_timing(steady_stretch, dt=1.0, transient=typical_step, max_window_s=3600)
print(s)
# block_s = 64 s, window_s = 2560 s (40 blocks); noise correlation time 21.3 s, process time constant 601 s
# - block: 3 x the noise correlation time 21.3 s, rounded to 64 samples, so block means are nearly independent
# - the block is at most 0.2 x the process time constant 601 s, so blocks follow the transient
# - window: 40 blocks = 2560 s; a steady call lags by about one window
# - measured lag-1 autocorrelation of block means on the steady stretch: 0.18
timing = s.timing(settle_s=300, hold_s=120)
```

When the noise needs longer blocks than the process allows, the suggestion
says so (a `CONFLICT` note) and keeps the blocks short enough to follow the
process. The drift test's coverage factor then grows, which is the honest
answer for such data. Averaging channels, or a quieter signal, is the
remedy.

```{eval-rst}
.. autofunction:: rocksteady.suggest_timing
   :no-index:
```

A detector whose window has fewer blocks than a criterion's calibration
needs (`recommended_blocks`: 20 for the drift and asymptote tests, 10 for
the band test) issues a `rocksteady.CalibrationWarning`. It still runs. If
the short window is a deliberate choice, filter the warning:
`warnings.filterwarnings("ignore", category=rs.CalibrationWarning)`.

## Checklist for a new adapter

- **Data:** sample times are seconds on a regular grid; resample first if
  needed (see `Timing.grid_tolerance`). Raw columns are named consistently.
- **Calibration:** noise models and limits come from data known to be
  steady. The deviation test's limit and noise are calibrated together (see
  {doc}`methods`).
- **Windows:** block and window lengths from `rs.suggest_timing` on a
  steady stretch (and a typical transient); about 40 or more blocks for the
  `ar1` drift test (see {doc}`drift`). Heed any `CalibrationWarning`.
- **Events:** logged actions are passed as an `EventLog` on the record's
  clock.
- **Validation:** labels are compared against a trusted reference, if one
  exists, before the adapter replaces it.

# Changelog

## Unreleased

### Added

- Subtypes for best-fit segments (#61, part of #55): `rs.Subtype(state,
  level=(lo, hi))` names a refinement of a state whose segments' mean lies
  in `[lo, hi)` (`None` for an open end), and `optimal_segments(...,
  subtypes={name: Subtype})` gives every segment of that state one of its
  subtypes (ADR 0001; a subtype with no conditions is a catch-all).
  `max_counts` caps subtypes per period as well as states, and both caps
  bind. Subtypes of one state may overlap; an exact tie goes to the one
  declared first (among segmentations of equal objective, the least sum of
  the subtypes' positions, which can also decide an exact tie between
  states). `refine` keeps each segment's subtype while it moves
  boundaries, then types its final segments again by the same rule (#66),
  and pruning stays exact.
  Segments carry `Segment.subtype` (None for a state without subtypes),
  `OptimalSegmentation.subtypes` holds the subtypes, `counts` has a column
  per subtype, and `summary()` lists counts by subtype. With no subtypes,
  results are unchanged.
- Peak and rise-time conditions for subtypes (#64, part of #55):
  `Subtype(..., peak=(lo, hi))` needs the segment's maximum in `[lo, hi)`
  (either end open), and `max_rise_s` with `rise_fraction` (default 0.95)
  needs the segment to reach `y0 + rise_fraction * (peak - y0)` within
  `max_rise_s` of its first sample, `y0` being its first finite sample, so
  rise time works for a non-zero baseline. A segment with no finite sample
  meets neither, and a segment must meet all of its subtype's conditions.
- Event conditions for subtypes (#65, part of #55): `optimal_segments(...,
  events={name: boolean mask})` takes named masks with one value per
  sample, and `Subtype(..., events={name: True or False})` needs the
  segment to contain, or not to contain, a flagged sample of that mask in
  `[start, stop)`, so that a driven transient is told apart from an
  undriven one, by actuator. There is no tolerance; widen a mask to accept
  an event just outside a segment. Every mask a condition names must be
  given, every mask must be boolean and as long as `t`, and a subtype
  names each mask once.
  `info["events"]` holds the masks.
- Every constraint of `optimal_segments` can name subtypes, and the
  record's first and last segments can be restricted (#63, part of #55).
  `transitions` names states or subtypes; a state's name stands for all of
  its subtypes, and the allowed pairs are the union over all entries, so
  the operating sequence (cold → heat-up → hot → cool-down → cold) tells
  apart transient subtypes with the same conditions by their neighbours.
  New `initial=` and `final=` sets of state or subtype names restrict the
  first and last segments (default: any; a one-segment record must satisfy
  both), so a transient cut off by the record's edge need not be called a
  startup. `min_duration_s` / `max_duration_s` take subtypes, and a
  subtype's and its state's bounds both bind; a subtype whose bounds leave
  no duration is refused up front. `penalty` takes subtypes: a
  subtype's value replaces its state's, and with `"bic"` (or when the
  mapping does not name it) a subtype takes its state's.
  `OptimalSegmentation.penalty` lists every state's and then every
  subtype's penalty. The infeasibility error names `initial` and `final`
  when they are given.
- `rs.classify_segments(t, y, segments, subtypes=..., events=...,
  transitions=..., initial=..., final=...)` gives subtypes to segments the
  analyst already has, such as the detector's, by the same definitions as
  `optimal_segments` (#66, part of #55). It runs the optimiser's dynamic
  program with the breakpoints and states fixed and no count caps,
  durations or penalties, so ties go to the least sum of the subtypes'
  positions among their states' (with no transitions, `initial` or
  `final`, each segment's first declared subtype it meets) and a transient
  with no conditions is typed from its neighbours, as in the optimiser.
  On the optimiser's own segments, by any method, it returns the
  optimiser's subtypes when those constraints name states only. It
  returns the segments with `subtype` set (None for a state without
  subtypes), and when they cannot be typed it raises a `ValueError` that
  names where, as the optimiser's infeasibility error does.
- The error for an infeasible `optimal_segments` also names the subtypes'
  conditions (#61).
- `optimal_segments` names the argument and the valid states when `models`
  names an unknown state (#61).

### Changed

- A `penalty` mapping for `optimal_segments` that names something other
  than a state with a model or a subtype is now rejected, like every other
  constraint; it used to be ignored (#63).
- When no segmentation satisfies the constraints, the `ValueError` from
  `optimal_segments` names the earliest candidate time after which none can
  continue, the period containing it (its start, as `count_by_period`
  indexes periods, or "the whole record"), and the caps and transitions in
  force, so the unusual day in a long batch run can be found (#62).
- `optimal_segments` no longer fits the model of a state capped at 0
  segments per period, which can never be used, so `n_fits` can be lower;
  the results are unchanged. With pruning, an infeasible call fits models
  only up to where every segmentation gets stuck (#62).

### Fixed

- `count_by_period` and `OptimalSegmentation.counts` build their table as
  integers, so pandas 2.2 and 2.3 no longer warn about downcasting on
  `fillna`; the tables are unchanged. On pandas 1.5, a table with no
  segments and a `DatetimeIndex` no longer fails, and an empty one with
  numeric periods has a float index (it was object), as on later pandas (#75).

### Documentation

- The docs and docstrings follow the vocabulary in `GLOSSARY.md`, now
  published as a glossary page with ADR 0001 (subtypes) and the
  contributor-agent notes (`AGENTS.md`, `docs/agents/`) (#56). Wording only;
  no API names change:
  - a contiguous run in one state is a *segment*; the "Intervals" guide
    page is now "Segments, fits and statistics by state";
  - the optimiser has *constraints* (counts per period, transitions,
    durations, penalties); a *limit* is a criterion's threshold;
  - overrides are domain knowledge that forces a label, not "rules";
  - the spacing of samples is the *sample spacing*; a *period* is the
    optimiser's counting span.
- Error messages say "sample spacing" for `Timing.sample_period_s`, and an
  infeasible `optimal_segments` reports that no segmentation satisfies the
  *constraints* (#56).
- `Report.provenance["parameter_reference"]` says "segment centre" for
  `poly` (was "interval centre"), so exported JSON reads differently (#56).
- `docs/agents/issue-tracker.md` gives the issue format, and what a child
  ticket adds: the sub-issue link, a row in its parent's table, `blocked_by`
  links (#87).
- `docs/agents/review.md`: probe boundaries and tie-breaks with exact
  arithmetic on quantised data; check that a PR is mergeable before
  waiting for CI, and test-merge it with the current base before merging.
  The issue format asks for boundaries and tie-breaks as checkable
  properties (#93).
- `docs/agents/review.md`: probe restatements. For each argument or rule a
  change alters, grep its name in the module docstring, `README.md`,
  `docs/source/` and the error messages, and check that each sentence is
  still true (#106).
- `docs/agents/review.md`: probe with mutants. Break each added line once,
  besides disabling each guard: drop each coercion, swap each reduction for
  its neighbour, shift each index or bound by one; some test must go red.
  Wait for CI only once `headRefOid` is the pushed SHA, as `mergeable` can
  still describe the previous head; read checks with `--json name,bucket`
  (#111).
- `docs/agents/coding-standards.md` holds the judgement calls a reviewer
  checks in every diff, and names the checks that decide the mechanical
  rules; `docs/agents/review.md` points at it. It starts with two rules from
  PR #104: a test pins the exact value a conversion produces, and a widened
  input brings its degenerate cases (#92).

### Tests and CI

- A `docs` extra (`pip install -e ".[docs]"`, included in `dev`) replaces
  `docs/requirements.txt` for CI and Read the Docs (#73).
- `tools/check.sh` runs CI's fast checks locally: the tests without `slow`,
  and the strict docs build (#73).
- `tests/test_vocabulary.py` fails on phrases `GLOSSARY.md` avoids that are
  always wrong (#73).
- Each avoided phrase in `tests/test_vocabulary.py` carries a near-miss,
  correct text the scan must not flag (#84).
- `tools/check.sh` prints the Python it runs, and stops early, naming the
  fix, when that Python is missing or lacks the dev extra (#84).
- `tools/check.sh <pytest args>` runs only that test selection, without
  the docs build, and needs only the test extra (#87).
- mypy checks `src/rocksteady` (configured in `pyproject.toml`; in the
  `typecheck` extra, included in `dev`) in a new CI job and in the full
  `tools/check.sh` run. The 62 errors it found are fixed with
  annotations, narrowing, renamed variables and equivalent rewrites
  (`np.max(x)` for `x.max()`, for example); no behaviour changes (#93).
- `pyproject.toml` configures pylint for editors that lint with it:
  120-character lines, and no notices for the docstring, size and naming
  conventions the code does not follow (#93).
- When its interpreter lacks a module, `tools/check.sh` names the
  distributions to install (`pip install mypy`) instead of
  `pip install -e ".[dev]"`, which, run from a worktree, would point the
  environment's editable install at the worktree (#93).
- The sdist ships every file under `tests/` (`MANIFEST.in` grafts the
  directory and excludes compiled files), so test data of a new file type
  needs no listing; CI's `build` job checks the sdist's `tests/` against
  `git ls-files tests` (#75).
- A `FutureWarning` fails the test that raises it (`filterwarnings` in
  `pyproject.toml`) (#75).
- pytest imports the checkout's own `src/` (`pythonpath` in
  `pyproject.toml`; the `test` extra needs `pytest>=7`), even where the
  environment's editable install points at another checkout, and so do
  the processes tests start (`tests/conftest.py` puts the same
  `rocksteady` first on `PYTHONPATH`). To run an unpacked sdist's tests
  against an installed package, pass `-o pythonpath=`, as CI's `build`
  job does; a test fails when an sdist's tests import its own `src/` (#99).
- `tools/lint.sh` runs pylint on four messages only: unused imports,
  unused variables, self-assignment and lambda assignment, over `src`,
  `tests`, `tools` and `examples`. The full `tools/check.sh` run and a new
  `lint` CI job run it (a `lint` extra, included in `dev`). The 16 hits it
  found are fixed, one an unused import in `rocksteady.report` (#106).
- `tests/test_wording_of_tests.py` fails on a test comment or docstring that
  names a review or a pull request by number (`review 1 of PR #83`,
  `PR #46`, `pull request #102`); a test says what it pins down. The
  existing ones are reworded (#106).
- `tools/lint.sh` also runs pycodestyle on blank lines (E301, E302, E303,
  E305, E306): two around a top-level definition, one before a method or a
  nested definition, none in excess. pylint does not check these. Both
  linters always run, so one run reports every finding. `pycodestyle` joins
  the `lint` extra, the three hits it found are fixed, and CI's `lint` job
  also runs `tests/test_lint_sh.py` (#111).
- `tests/test_frozen_values.py`: every public frozen value (`Signal`,
  `Timing`, `Subtype`, `Segment`), with every optional field set, rebuilds
  equal and hashes alike through `dataclasses.replace`, its constructor
  given every field, and pickle. A public frozen dataclass without a case
  fails it (#111).
- `tools/mutate.py SPEC.py` runs a mutation probe (#101). A Python spec
  names the file, the pytest selection and the mutants (literal `old` →
  `new`, each `old` matching exactly once, or the spec is refused before
  any test runs). The tool runs the selection once per mutant and restores
  the file byte for byte after each, also on Ctrl-C. It reports each mutant
  red or survived, and exits non-zero when an unmarked mutant survives.
  The file's cached bytecode is deleted first, so a same-length mutant
  cannot run as the cached original. `docs/agents/review.md` points the
  mutant probe at it.

## 0.5.0

Best-fit segments under limits (#45), the documentation site (#47, #49), the
asymptote criterion (#7), and data-driven timing (#51).

Upgrading from 0.4.0. Behaviour changes to check:
- **Breaking:** `Timing` requires `block_s` and `window_s`, and `Detector`
  requires a `timing` (#51). Code that relied on the old 30 s / 300 s
  defaults raises a clear error; `rs.suggest_timing` proposes values.
- A detector whose window has fewer blocks than a criterion's calibration
  needs issues a `CalibrationWarning` (#51).
- Criteria receive the times at which block means were measured. This
  differs from 0.4.0 only for blocks with tolerated missing samples (#7).
- `batch.run(with_results=True)` adds `detail_<criterion>_<key>` columns for
  criteria that declare diagnostics (#7).

### Added

- `rs.suggest_timing(steady, dt, transient=None, ...)` proposes `block_s`
  and `window_s` from the application's data (#51). It measures:
  - the noise correlation time of a steady stretch, for a block of a few
    correlation times;
  - optionally, a typical transient's time constant, which caps the block
    length.

  Missing samples keep their place on the time grid: correlations are
  measured within contiguous runs, never across a gap. The window is 40
  blocks unless `max_window_s` limits it, and never exceeds that limit. It
  returns a `TimingSuggestion` with the measurements and the reasoning
  (including conflicts), and `.timing(...)` builds a `Timing`. It sets
  nothing itself.
- `rocksteady.CalibrationWarning` and `Criterion.recommended_blocks`: a
  detector whose window has fewer blocks than a criterion's calibration needs
  warns. The thresholds are 20 for `DriftCriterion` and `AsymptoteCriterion`
  and 10 for `BoundCriterion` (#51).

- `rs.AsymptoteCriterion` (`"asymptote"` in specs): certifies that a
  first-order approach has reached its asymptote, i.e. that the remaining
  approach `d = A exp(-t_end / tau)` at the window end is within a limit
  (#7).
  - **The test** takes the worst case over every `tau` the data allow (a
    profile-likelihood confidence set on a `tau` grid up to the caller's
    `tau_max`), each `d(tau)` with an AR(1) uncertainty. A test that trusts
    a fitted `tau` passed falsely up to 15 % of the time in calibration
    when the window spanned about `tau`.
  - **Flat or unresolved windows** are bounded by about `tau_max × |slope|`,
    with no separate mode.
  - **Calibration:** at the limit, with `tau_max` = 2 tau, the false-pass
    rate is at most 2.5 % for 40 or more blocks and about 5 % at 20
    (nominal 2.5 %); `tools/coverage_mc.py --asymptote` reproduces it.
  - **Cost:** about 1 ms per evaluation (1.5 ms with 300-sample blocks).
- `CriterionResult.details` (optional numeric diagnostics) and
  `Criterion.detail_keys`. `batch.run(with_results=True)` adds
  `detail_<criterion>_<key>` columns, the same for an empty record, and
  rejects two diagnostics that would share a column name, before reading any
  input (#7).
- `Criterion.uses_block_masks`: a criterion that sets it receives
  `valid`, the (n_blocks, block_samples) validity of the samples behind
  each block mean (None when all blocks are complete).
  `AsymptoteCriterion` averages its exponential basis over exactly those
  samples, which is exact for any pattern of tolerated gaps (#7).
- `Criterion.bind_timing(timing)`: an optional hook the `Detector` calls once
  with its `Timing`. `AsymptoteCriterion` uses it to evaluate at the current
  sample, `(block_samples - 1) * sample_period_s / 2` after the last block
  centre (#7).
- `fit.golden_refine`: the golden-section refinement shared by the fit
  models and `AsymptoteCriterion`; fit results are unchanged (#7).
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

### Changed

- **Breaking:** `Timing` requires `block_s` and `window_s`, and `Detector`
  requires a `timing`. The old placeholders (30 s blocks in a 300 s window,
  10 blocks) were in the range the docs call unreliable for the drift test.
  The error message points to `rs.suggest_timing` and the new
  "Choosing block and window lengths" guide section. `sample_period_s` keeps
  its default of 1 s, since a wrong one shows up at once as rejected or
  gapped timestamps (#51).
- Criteria receive the times at which the block means were measured. A
  block with tolerated missing samples is placed at the mean time of its
  valid samples rather than at its nominal centre; complete blocks are
  unchanged. This corrects the drift slope's timing on partial blocks, and
  keeps `AsymptoteCriterion` from extrapolating a partial block's mean
  (#7).

### Fixed

- `DeviationCriterion(relative=True)` on a window whose mean and
  `scale_floor` are both zero is undecided; it raised `ZeroDivisionError`
  (#7).

### Documentation

- An adapter guide and a methods page on the documentation site (#49).
  - **Writing an adapter:** the four-part pattern (configuration, a
    detector from a spec, streaming, batch), the signatures of the calls
    adapters make (rendered from the code), and the contracts and
    registration of custom criteria, disturbances, overrides, slope
    methods and fit models.
  - **Two runnable examples** in `docs/source/examples/`. The test suite
    runs them (`tests/test_docs_examples.py`), and the sdist now includes
    `docs/`.
  - **Methods:** the mathematics as implemented, with links to each
    function. It covers block means, the three-way equivalence test and the
    margin ratio, the deviation, drift (`ar1`, `nw`, with the calibration
    table) and band criteria, the integrated autocorrelation time and
    `N_eff`, disturbances and the state machine, fit models with variable
    projection and the AR(1) sandwich covariance, the comparison statistics,
    and constrained optimal segmentation by dynamic programming.
  - **The README's references** are now alphabetical and extended (Bellman;
    Golub and Pereyra; Jackson et al.; JCGM 100; Neumaier; Schuirmann;
    Schwarz; Sokal; Truong et al.; Wellek; White), still the single
    bibliography.
- A Read the Docs site, https://rocksteady.readthedocs.io (#47).
  - **Build:** Sphinx with the Read the Docs theme, configured by
    `.readthedocs.yaml`, `docs/requirements.txt` and `docs/source/conf.py`.
  - **Guide:** the pages include the README's sections, so the README stays
    the single source.
  - **API reference:** one page per public module, generated from the
    docstrings. The `Timing` and `Reason` docstrings and a few others were
    reformatted as valid reStructuredText.
  - **CI:** a `docs` job builds with warnings as errors.
  - **README:** links to the site; its example, changelog and licence links
    are now absolute.

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

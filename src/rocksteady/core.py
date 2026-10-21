"""
Generic steady-state detection.

A detector is assembled from five kinds of parts. Each experiment supplies
its own instances; the state machine that combines them is shared.

1. **Signals**: named scalars computed from each raw sample (a column, or a
   derived quantity such as a channel mean or a spread).
2. **Criteria**: tests applied to one signal over the trailing window. Each
   returns a statistic, its uncertainty and a limit, from which the detector
   forms a margin ratio and a three-way verdict (pass, fail, undecided).
3. **Disturbances**: per-sample detectors of external action, either found
   in the signals (a discrete actuator move) or supplied from a log
   (control or configuration events). A disturbance resets the clocks.
4. **Timing**: block length, window length, settling time after a
   disturbance, hold time before acceptance, and whether an undecided
   window keeps the last decided label.
5. **Overrides**: domain knowledge that can force a label (for example, a
   system shut down below its operating floor is steady by definition).

Labels
------
- DISTURBED: a disturbance is active now.
- SETTLING: within the quiet time after a disturbance, or at least one
  criterion fails beyond its uncertainty.
- INDETERMINATE: not enough history, or the uncertainty straddles a limit
  and no earlier decision is being held.
- HOLDING: all criteria pass, but for less than the hold time.
- STEADY: all criteria pass and the hold time has elapsed.

Each verdict also carries a ``Reason`` saying why it has its label (a
window certified by data, a forced label, missing data, low block
coverage, too little history, and so on).

Everything but STEADY (and HOLDING, which is steady awaiting confirmation)
is "transient" in the two-class sense. The detector is causal; batch
evaluation of a record gives exactly the online labels.

This module holds the contracts only and depends on numpy alone. Concrete
criteria, disturbances, overrides and signal builders live in their own
modules and are registered by name in ``rocksteady.registry``.
"""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from . import _validate as _v


class Label(str, Enum):
    DISTURBED = "disturbed"
    SETTLING = "settling"
    INDETERMINATE = "indeterminate"
    HOLDING = "holding"
    STEADY = "steady"

    def __str__(self) -> str:
        return self.value


class Reason(str, Enum):
    """
    Why a verdict has its label.

    - ``CERTIFIED``: all criteria pass on the window (HOLDING or STEADY).
    - ``FORCED``: an override forced the label (``Verdict.forced_by``).
    - ``DISTURBANCE``: a disturbance is active (DISTURBED).
    - ``SETTLING_TIME``: the quiet time after a disturbance, or at the start
      of a record, has not yet elapsed.
    - ``FAILED``: at least one criterion fails beyond its uncertainty.
    - ``UNDECIDED``: statistical overlap; the uncertainty straddles a limit.
    - ``INVALID_RESULT``: a criterion returned an unusable result
      (``CriterionResult.invalid_reason``).
    - ``LOW_COVERAGE``: some block has too few valid samples
      (``Timing.min_block_coverage``).
    - ``INSUFFICIENT_HISTORY``: fewer samples than one window since the start
      or the last reset.
    - ``MISSING``: this sample is non-finite, and the gap is tolerated.
    - ``UNREADABLE``: a signal could not be computed from this sample (a
      missing key or a bad value), and the gap is tolerated.
    - ``GAP_RESET``: missing data (this sample, or a jump in time before it)
      exceeded ``Timing.max_gap_s`` and cleared the history.

    The last three label a sample INDETERMINATE whatever the hold policy.
    UNDECIDED, INVALID_RESULT and LOW_COVERAGE may keep the last decided
    label when ``Timing.hold_when_undecided`` is set.
    """

    CERTIFIED = "certified"
    FORCED = "forced"
    DISTURBANCE = "disturbance"
    SETTLING_TIME = "settling_time"
    FAILED = "failed"
    UNDECIDED = "undecided"
    INVALID_RESULT = "invalid_result"
    LOW_COVERAGE = "low_coverage"
    INSUFFICIENT_HISTORY = "insufficient_history"
    MISSING = "missing"
    UNREADABLE = "unreadable"
    GAP_RESET = "gap_reset"

    def __str__(self) -> str:
        return self.value


#: Reasons for a tolerated missing sample. The hold clock pauses over them,
#: and ``states.segments`` bridges them inside a steady run.
MISSING_REASONS = frozenset({Reason.MISSING, Reason.UNREADABLE})


class Outcome(str, Enum):
    """Three-way result of one criterion on one window."""

    PASS = "pass"
    FAIL = "fail"
    UNDECIDED = "undecided"


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

Source = Union[str, Callable[[Mapping[str, float]], float]]


@dataclass(frozen=True)
class Signal:
    """
    A named scalar derived from each raw sample.

    ``source`` is a column name or a callable taking the raw sample mapping.
    """

    name: str
    source: Source

    def __post_init__(self):
        _v.name(self.name, "Signal name")
        if not (isinstance(self.source, str) and self.source) and not callable(self.source):
            raise ValueError(f"Signal {self.name!r}: source must be a column name or a callable, got {self.source!r}")

    def value(self, sample: Mapping[str, float]) -> float:
        if isinstance(self.source, str):
            return float(sample[self.source])
        return float(self.source(sample))


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------


@dataclass
class CriterionResult:
    """
    Outcome of one criterion on one window.

    The true value of the statistic is taken to lie within
    ``statistic +/- k * uncertainty``. Comparing that interval with the limit
    gives the three-way test.

    ``details`` holds optional numeric diagnostics (for example a fitted
    time constant). A criterion that fills it declares the keys in
    ``Criterion.detail_keys``, so ``batch.run(with_results=True)`` can give
    them columns; they never affect the outcome.
    """

    name: str
    statistic: float
    uncertainty: float
    limit: float
    k: float
    details: Dict[str, float] = field(default_factory=dict)

    @property
    def upper(self) -> float:
        return self.statistic + self.k * self.uncertainty

    @property
    def lower(self) -> float:
        return self.statistic - self.k * self.uncertainty

    @property
    def invalid_reason(self) -> Optional[str]:
        """
        Why this result cannot decide anything, or None. A criterion or a
        callable noise model may return values no window can be judged on:
        non-finite numbers, a negative uncertainty or coverage factor, or a
        limit that is not positive.
        """
        for label, value in (("statistic", self.statistic), ("uncertainty", self.uncertainty),
                             ("limit", self.limit), ("k", self.k)):
            if not np.isfinite(value):
                return f"{label} is not finite ({value!r})"
        if self.uncertainty < 0:
            return f"uncertainty is negative ({self.uncertainty!r})"
        if self.k < 0:
            return f"coverage factor k is negative ({self.k!r})"
        if not self.limit > 0:
            return f"limit is not positive ({self.limit!r})"
        return None

    @property
    def ratio(self) -> float:
        """
        Margin ratio: below 1 passes with margin, above 1 is not demonstrated.

        Infinite for an invalid result (see ``invalid_reason``), so the worst
        ratio over several criteria does not depend on their order.
        """
        if self.invalid_reason is not None:
            return np.inf
        r = self.upper / self.limit
        return np.inf if np.isnan(r) else float(r)

    @property
    def outcome(self) -> Outcome:
        """PASS, FAIL or UNDECIDED; an invalid result is always UNDECIDED, never a pass."""
        if self.invalid_reason is not None:
            return Outcome.UNDECIDED
        if self.upper <= self.limit:
            return Outcome.PASS
        if self.lower > self.limit:
            return Outcome.FAIL
        return Outcome.UNDECIDED


class CalibrationWarning(UserWarning):
    """
    A detector is configured outside the range in which a criterion's
    uncertainty is calibrated (for example, too few blocks per window). It
    still runs; filter the warning with ``warnings.filterwarnings`` if the
    choice is deliberate.
    """


class Criterion(ABC):
    """
    A test on one signal over the trailing window of block means.

    ``min_blocks`` is the fewest blocks the criterion can decide on; the
    detector refuses a window shorter than the largest ``min_blocks``.
    ``recommended_blocks`` (optional) is the fewest at which its uncertainty
    is calibrated; the detector issues a ``CalibrationWarning`` below it.
    ``detail_keys`` names the entries the criterion puts in
    ``CriterionResult.details`` (none by default).

    ``bind_timing(timing)`` is called once by the ``Detector`` that uses the
    criterion, before any data, for a criterion that needs more than the
    block times, such as where the current sample lies relative to the last
    block centre. The default does nothing.

    A criterion with ``uses_block_masks = True`` is called as
    ``evaluate(t_blocks, y_blocks, valid=mask)``, where ``mask`` is the
    (n_blocks, block_samples) validity of the samples behind each block mean,
    or None when every block is complete. A nonlinear model needs it: a
    block's time alone does not say which samples its mean averages.
    """

    name: str
    signal: str
    min_blocks: int = 3
    recommended_blocks: Optional[int] = None
    detail_keys: Tuple[str, ...] = ()
    uses_block_masks: bool = False

    @abstractmethod
    def evaluate(self, t_blocks: np.ndarray, y_blocks: np.ndarray) -> CriterionResult:
        """
        ``y_blocks`` are the block means; ``t_blocks`` the times they were
        measured at, in seconds from the window's centre: the block centres
        for complete blocks, the mean time of the valid samples for a block
        with tolerated missing samples.
        """

    def bind_timing(self, timing: "Timing") -> None:
        """Called once by the ``Detector``, with its ``Timing``; override to use it."""


# ---------------------------------------------------------------------------
# Disturbances and overrides
# ---------------------------------------------------------------------------


class Disturbance(ABC):
    """
    Per-sample detector of external action. ``update`` returns True while active.

    Disturbances (and overrides) may keep state between samples, which the
    detector clears with ``reset``. An instance belongs to one detector:
    two detectors sharing it, used sample by sample in turn, would update
    and reset each other's state. (Running them one after the other with
    ``batch.run``, which resets first, is safe.) Build separate instances
    for each detector, for example by calling ``build_detector(spec)`` once
    per detector; rocksteady never copies components for you.
    """

    name: str

    @abstractmethod
    def update(self, t: float, sample: Mapping[str, float], values: Mapping[str, float]) -> bool:
        ...

    def reset(self) -> None:
        pass


@dataclass
class Context:
    """What an override may look at when deciding to force a label."""

    t: float
    sample: Mapping[str, float]
    values: Mapping[str, float]
    disturbed: bool
    disturbance_onset: bool
    since_disturbance_s: float


class Override(ABC):
    """
    Domain knowledge that may force a label. Return None to defer to the detector.

    Like a ``Disturbance``, an instance may keep state and belongs to one
    detector.

    ``apply`` is called for every readable, finite sample, disturbed ones
    included (a disturbance still takes precedence over a forced label). A
    sample with a missing, non-finite or unreadable signal is labelled
    INDETERMINATE without calling it, and a gap that exceeds
    ``Timing.max_gap_s`` resets the override with the history.
    """

    name: str

    @abstractmethod
    def apply(self, ctx: Context) -> Optional[Label]:
        ...

    def reset(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Timing and verdicts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Timing:
    """
    Time scales of the detector.

    Parameters
    ----------
    sample_period_s
        Spacing of the samples fed to ``update``.
    block_s
        Block length at which criteria are evaluated. It is rounded to a
        whole number of samples (``effective_block_s``) and must be at least
        one sample.
    window_s
        Trailing window length, rounded to whole samples. The criteria
        evaluate its last ``n_blocks`` whole blocks (``evaluated_window_s``).
        With a window that is not a whole number of blocks, the leading
        remainder is history the detector waits for but does not evaluate.
    settle_s
        Quiet time required after a disturbance before a window is
        evaluated, in addition to the window itself.
    hold_s
        Continuous passing time before HOLDING becomes STEADY.
    hold_when_undecided
        An undecided window (statistical overlap, an invalid criterion
        result, or low block coverage) keeps the last decided label instead
        of returning INDETERMINATE.
    undecided_counts_toward_hold
        When an undecided window keeps a passing state, it also advances the
        hold clock. If False, the hold clock pauses until a window passes
        again.
    max_gap_s
        Longest run of missing data that is tolerated without clearing the
        history. Missing data are non-finite or unreadable samples, and the
        samples implied by a jump in ``t`` of more than 1.5 sample spacings.
        With 0 (the default) any missing data resets the detector.
    min_block_coverage
        When gaps are tolerated, the smallest fraction of valid samples
        every block in the window must have for the criteria to be
        evaluated; otherwise the window is undecided.
    grid_tolerance
        Sampling contract. Times passed to ``update`` must lie on a grid of
        ``sample_period_s``: each step must be within ``grid_tolerance``
        sample spacings of a whole number (at least one) of spacings. A step
        of k spacings (k >= 2) is a gap of k - 1 missing samples. The default,
        0.5, accepts any step of at least half a spacing and rounds it to the
        grid; a smaller value rejects jittery or drifting clocks. Steps
        shorter than ``1 - grid_tolerance`` spacings (sampling faster than
        ``sample_period_s``) are always rejected: resample the record first.

    Notes
    -----
    Missing data and the hold clock: a tolerated missing sample is labelled
    INDETERMINATE (reason MISSING or UNREADABLE) whatever the two undecided
    options, because no window is evaluated on it. The hold clock *pauses*:
    it neither resets nor advances, for missing samples and for omitted
    timestamps alike, so after a tolerated gap a passing run resumes where
    it stopped and STEADY comes ``hold_s`` of passing windows after the
    first, plus the gap. ``states.segments`` bridges such samples inside a
    steady run, so a gap given as NaN rows and the same gap given as
    omitted timestamps yield the same steady segments. The quiet time
    after a disturbance does advance over a tolerated gap.
    """

    sample_period_s: float = 1.0
    block_s: Optional[float] = None
    window_s: Optional[float] = None
    settle_s: float = 0.0
    hold_s: float = 0.0
    hold_when_undecided: bool = False
    undecided_counts_toward_hold: bool = True
    max_gap_s: float = 0.0
    min_block_coverage: float = 0.5
    grid_tolerance: float = 0.5

    def __post_init__(self):
        missing = [k for k in ("block_s", "window_s") if getattr(self, k) is None]
        if missing:
            raise ValueError(
                f"Timing needs {' and '.join(missing)}: block and window lengths depend on the application's "
                "noise and time scales, and there is no safe default. rs.suggest_timing(steady, dt) proposes "
                "them from data known to be steady; see 'Choosing block and window lengths' in the docs.")
        _v.finite(self.sample_period_s, "Timing.sample_period_s", positive=True)
        _v.finite(self.block_s, "Timing.block_s", positive=True)
        _v.finite(self.window_s, "Timing.window_s", positive=True)
        for opt in ("settle_s", "hold_s", "max_gap_s"):
            _v.finite(getattr(self, opt), f"Timing.{opt}", nonnegative=True)
        _v.finite(self.min_block_coverage, "Timing.min_block_coverage")
        if not 0 < self.min_block_coverage <= 1:
            raise ValueError("min_block_coverage must be in (0, 1]")
        _v.finite(self.grid_tolerance, "Timing.grid_tolerance")
        if not 0 < self.grid_tolerance <= 0.5:
            raise ValueError(f"grid_tolerance must be in (0, 0.5] sample spacings, got {self.grid_tolerance!r}")
        if self.block_s < self.sample_period_s * (1 - 1e-9):
            raise ValueError(f"block_s ({self.block_s}) must be at least one sample spacing ({self.sample_period_s})")

    @property
    def window_samples(self) -> int:
        return int(round(self.window_s / self.sample_period_s))

    @property
    def block_samples(self) -> int:
        return max(1, int(round(self.block_s / self.sample_period_s)))

    @property
    def n_blocks(self) -> int:
        return self.window_samples // self.block_samples

    @property
    def effective_block_s(self) -> float:
        """Block duration actually used: a whole number of samples (``block_s`` rounded)."""
        return self.block_samples * self.sample_period_s

    @property
    def evaluated_samples(self) -> int:
        """Samples the criteria evaluate: the last ``n_blocks`` whole blocks of the window."""
        return self.n_blocks * self.block_samples

    @property
    def evaluated_window_s(self) -> float:
        """Duration the criteria evaluate (the certifying window)."""
        return self.evaluated_samples * self.sample_period_s

    @property
    def effective(self) -> Dict[str, float]:
        """The timing as the detector applies it, after rounding to whole samples."""
        return {"sample_period_s": self.sample_period_s, "block_s": self.effective_block_s,
                "block_samples": self.block_samples, "n_blocks": self.n_blocks,
                "window_s": self.window_samples * self.sample_period_s, "window_samples": self.window_samples,
                "evaluated_window_s": self.evaluated_window_s, "evaluated_samples": self.evaluated_samples}

    @property
    def quiet_samples(self) -> int:
        """Samples since the last disturbance before a window may be evaluated."""
        return max(self.window_samples, int(round((self.window_s + self.settle_s) / self.sample_period_s)))

    @property
    def hold_samples(self) -> int:
        return int(round(self.hold_s / self.sample_period_s))


@dataclass
class Verdict:
    """Result of one ``update``."""

    t: float
    label: Label
    results: Dict[str, CriterionResult] = field(default_factory=dict)
    since_disturbance_s: float = 0.0
    forced_by: Optional[str] = None
    reason: Optional[Reason] = None
    valid: bool = True  # the sample was readable and finite, and is in the detector's history
    window_start_s: float = float("nan")  # time of the first evaluated sample of the window examined, if any

    @property
    def eta(self) -> float:
        """Worst margin ratio over the criteria."""
        return max((r.ratio for r in self.results.values()), default=np.nan)

    @property
    def limiting(self) -> Optional[str]:
        """Name of the criterion with the worst margin ratio."""
        if not self.results:
            return None
        return max(self.results.values(), key=lambda r: r.ratio).name

    @property
    def is_steady(self) -> bool:
        return self.label in (Label.STEADY, Label.HOLDING)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


def _duplicates(names) -> list:
    seen, dupes = set(), set()
    for n in names:
        (dupes if n in seen else seen).add(n)
    return sorted(dupes)


class Detector:
    """
    Streaming steady-state detector. Feed one sample per ``timing.sample_period_s``.

    Parameters
    ----------
    signals : sequence of Signal
    criteria : sequence of Criterion
        Each refers to a signal by name. Every criterion must pass for the
        window to pass.
    disturbances : sequence of Disturbance, optional
    overrides : sequence of Override, optional
        Applied in order after the disturbance check; the first non-None
        label wins.
    timing : Timing

    The detector owns its disturbances and overrides: it updates and resets
    their state, so an instance must not be shared with another detector
    in use at the same time (see ``Disturbance``).

    Times passed to ``update`` must increase and lie on the sampling grid
    (``Timing.grid_tolerance``). A step of more than 1.5 sample spacings is
    a gap; see ``Timing.max_gap_s``.
    """

    def __init__(
        self,
        signals: Sequence[Signal],
        criteria: Sequence[Criterion],
        disturbances: Sequence[Disturbance] = (),
        overrides: Sequence[Override] = (),
        timing: Optional[Timing] = None,
    ):
        if timing is None:
            raise ValueError("Detector needs a timing: rs.Timing(block_s=..., window_s=...); see rs.suggest_timing")
        self.signals = list(signals)
        self.criteria = list(criteria)
        self.disturbances = list(disturbances)
        self.overrides = list(overrides)
        self.timing = timing
        for kind, items in (("signal", self.signals), ("criterion", self.criteria)):
            dupes = _duplicates(x.name for x in items)
            if dupes:
                hint = " (give each criterion a distinct name=)" if kind == "criterion" else ""
                raise ValueError(f"duplicate {kind} names {dupes}{hint}")
        for c in self.criteria:
            _v.name(getattr(c, "name", None), f"{type(c).__name__} name")
        names = {s.name for s in self.signals}
        for kind, items in (("criterion", self.criteria), ("disturbance", self.disturbances),
                            ("override", self.overrides)):
            for item in items:
                ref = getattr(item, "signal", None)
                if (kind == "criterion" or ref is not None) and ref not in names:
                    raise ValueError(f"{kind} {getattr(item, 'name', type(item).__name__)!r} refers to unknown "
                                     f"signal {ref!r}; signals are {sorted(names)}")
        for c in self.criteria:
            bind = getattr(c, "bind_timing", None)
            if callable(bind):
                bind(timing)
        need = max([3] + [getattr(c, "min_blocks", 3) for c in self.criteria])
        if timing.n_blocks < need:
            raise ValueError(f"window_s must span at least {need} blocks for these criteria "
                             f"(it spans {timing.n_blocks})")
        for c in self.criteria:
            rec = getattr(c, "recommended_blocks", None)
            if rec is not None and timing.n_blocks < rec:
                warnings.warn(f"{type(c).__name__} {c.name!r}: the window holds {timing.n_blocks} blocks; at least "
                              f"{rec} are recommended for its calibration (see the calibration tables in the "
                              "docs, and rs.suggest_timing)", CalibrationWarning, stacklevel=2)
        self._index = {s.name: i for i, s in enumerate(self.signals)}
        self._n = timing.window_samples
        self._b = timing.block_samples
        self._nb = timing.n_blocks
        # Block centre times use the effective (whole-sample) block duration, the spacing of the block means.
        self._t_blocks = (np.arange(self._nb) - (self._nb - 1) / 2.0) * timing.effective_block_s
        self.reset()

    def reset(self) -> None:
        """Clear history (after a data gap)."""
        self._buf = np.full((self._n, len(self.signals)), np.nan)
        self._head = 0
        self._count = 0
        self._since_disturbance = 0
        self._was_disturbed = False
        self._seen_disturbance = False
        self._last_decided: Optional[bool] = None  # True = passing
        self._pass_run = 0
        self._gap_run = 0
        self._last_t: Optional[float] = None
        for d in self.disturbances:
            d.reset()
        for o in self.overrides:
            o.reset()

    # -- helpers ----------------------------------------------------------

    def _push(self, row: np.ndarray) -> None:
        self._buf[self._head] = row
        self._head = (self._head + 1) % self._n
        self._count += 1

    def _tolerate_gap(self, n_missing: int) -> bool:
        """
        Record ``n_missing`` missing samples as NaN rows if the gap is within
        ``max_gap_s``; otherwise reset. Returns True if the gap was tolerated.
        """
        run = self._gap_run + n_missing
        if self._count == 0 or run * self.timing.sample_period_s > self.timing.max_gap_s:
            self.reset()
            return False
        nan_row = np.full(len(self.signals), np.nan)
        for _ in range(min(n_missing, self._n)):
            self._push(nan_row)
        self._count += max(n_missing - self._n, 0)
        self._since_disturbance += n_missing
        self._gap_run = run
        return True

    def _window(self) -> np.ndarray:
        """Window samples in time order, shape (n, n_signals)."""
        return np.concatenate((self._buf[self._head :], self._buf[: self._head]))

    def _blocks(self) -> Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
        """
        ``(t_blocks, block_means, valid)`` over the window, or None if some
        block has too few valid samples (``Timing.min_block_coverage``). The
        means have shape (n_blocks, n_signals). ``t_blocks`` is the mean time
        of each block's valid samples: the block centre for a complete block,
        moved towards the valid samples in a partial one, since that is where
        its mean was measured. ``valid`` is the (n_blocks, block_samples)
        sample validity, or None when every block is complete.
        """
        w = self._window()[self._n - self._nb * self._b :].reshape(self._nb, self._b, -1)
        valid = np.isfinite(w[:, :, 0])  # missing samples are whole NaN rows
        if valid.all():
            return self._t_blocks, w.mean(axis=1), None
        if (valid.mean(axis=1) < self.timing.min_block_coverage).any():
            return None
        slot = np.arange(self._b) - (self._b - 1) / 2.0  # sample offsets from the block centre, in samples
        shift = (valid * slot).sum(axis=1) / valid.sum(axis=1) * self.timing.sample_period_s
        return self._t_blocks + shift, np.nanmean(w, axis=1), valid

    def _record(self, passing: Optional[bool]) -> None:
        if passing is not None:
            self._last_decided = passing
        self._pass_run = self._pass_run + 1 if passing else 0

    def _passing_label(self) -> Label:
        """STEADY once ``hold_s`` has elapsed since the first passing window."""
        return Label.STEADY if (self._pass_run - 1) >= self.timing.hold_samples else Label.HOLDING

    # -- main entry -------------------------------------------------------

    def update(self, t: float, sample: Mapping[str, float]) -> Verdict:
        """Add one sample at time ``t`` (seconds) and return the verdict."""
        dt = self.timing.sample_period_s
        gap_reset = False
        if self._last_t is not None:
            step = t - self._last_t
            if not step > 0:
                raise ValueError(f"times must increase: t={t} follows t={self._last_t}")
            k = step / dt
            tol = self.timing.grid_tolerance
            if k < 1 - tol:
                raise ValueError(f"t={t} follows t={self._last_t} by {step:g} s, less than the sample spacing "
                                 f"({dt:g} s): the record is sampled faster than sample_period_s; resample it")
            if abs(k - round(k)) > tol:
                raise ValueError(f"t={t} follows t={self._last_t} by {k:.3g} sample spacings, off the sampling "
                                 f"grid by more than grid_tolerance={tol:g}")
            if step > 1.5 * dt:
                gap_reset = not self._tolerate_gap(int(round(step / dt)) - 1)

        missing = None
        try:
            values = {s.name: s.value(sample) for s in self.signals}
            row = np.fromiter(values.values(), dtype=float, count=len(values))
            if not np.all(np.isfinite(row)):
                missing = Reason.MISSING
        except (KeyError, TypeError, ValueError):
            missing = Reason.UNREADABLE
        if missing is not None:
            # No window is evaluated: the hold clock pauses (``_record`` is not called).
            tolerated = self._tolerate_gap(1)
            self._last_t = t
            since_s = self._since_disturbance * dt if tolerated else 0.0
            return Verdict(t, Label.INDETERMINATE, since_disturbance_s=since_s,
                           reason=missing if tolerated else Reason.GAP_RESET, valid=False)
        self._last_t = t
        self._gap_run = 0
        self._push(row)

        # Update every detector (no short-circuit) so their histories stay current.
        flags = [d.update(t, sample, values) for d in self.disturbances]
        disturbed = any(flags)
        onset = disturbed and not self._was_disturbed
        self._was_disturbed = disturbed
        self._seen_disturbance = self._seen_disturbance or disturbed
        self._since_disturbance = 0 if disturbed else self._since_disturbance + 1
        since_s = self._since_disturbance * dt

        # Overrides observe every readable, finite sample (they may keep state), but a
        # disturbance takes precedence over any label they force. Missing samples returned above.
        ctx = Context(t, sample, values, disturbed, onset, since_s)
        forced = [(o.name, o.apply(ctx)) for o in self.overrides]

        if disturbed:
            self._record(False)
            return Verdict(t, Label.DISTURBED, since_disturbance_s=0.0, reason=Reason.DISTURBANCE)

        for name, label in forced:
            if label is not None:
                passing = label in (Label.STEADY, Label.HOLDING)
                self._record(passing if label != Label.INDETERMINATE else None)
                if passing:
                    label = self._passing_label()
                return Verdict(t, label, since_disturbance_s=since_s, forced_by=name, reason=Reason.FORCED)

        if self._count < self._n:
            return Verdict(t, Label.INDETERMINATE, since_disturbance_s=since_s,
                           reason=Reason.GAP_RESET if gap_reset else Reason.INSUFFICIENT_HISTORY)
        if self._since_disturbance < self.timing.quiet_samples:
            if not self._seen_disturbance:
                # Start of a record: the settling wait applies, but nothing was disturbed.
                self._record(None)
                return Verdict(t, Label.INDETERMINATE, since_disturbance_s=since_s, reason=Reason.SETTLING_TIME)
            self._record(False)
            return Verdict(t, Label.SETTLING, since_disturbance_s=since_s, reason=Reason.SETTLING_TIME)

        evaluable = self._blocks()
        w0 = t - (self.timing.evaluated_samples - 1) * dt  # the window's history is on the sample grid
        results = {}
        if evaluable is None:
            outcomes = [Outcome.UNDECIDED]
            undecided = Reason.LOW_COVERAGE
        else:
            t_blocks, blocks, valid = evaluable
            evaluated = [c.evaluate(t_blocks, blocks[:, self._index[c.signal]], valid=valid)
                         if getattr(c, "uses_block_masks", False) else
                         c.evaluate(t_blocks, blocks[:, self._index[c.signal]]) for c in self.criteria]
            outcomes = [r.outcome for r in evaluated]  # every criterion counts, whatever its name
            results = {c.name: r for c, r in zip(self.criteria, evaluated)}
            invalid = any(r.invalid_reason is not None for r in evaluated)
            undecided = Reason.INVALID_RESULT if invalid else Reason.UNDECIDED

        if Outcome.FAIL in outcomes:
            self._record(False)
            return Verdict(t, Label.SETTLING, results, since_s, reason=Reason.FAILED, window_start_s=w0)
        if all(o == Outcome.PASS for o in outcomes):
            self._record(True)
            return Verdict(t, self._passing_label(), results, since_s, reason=Reason.CERTIFIED, window_start_s=w0)
        # Undecided.
        if self.timing.hold_when_undecided and self._last_decided is not None:
            if not (self._last_decided and not self.timing.undecided_counts_toward_hold):
                self._record(self._last_decided)
            label = self._passing_label() if self._last_decided else Label.SETTLING
            return Verdict(t, label, results, since_s, reason=undecided, window_start_s=w0)
        self._record(None)
        return Verdict(t, Label.INDETERMINATE, results, since_s, reason=undecided, window_start_s=w0)


__all__ = [
    "Label", "Reason", "MISSING_REASONS", "Outcome", "Signal", "Criterion", "CriterionResult", "Disturbance",
    "Override",
    "Context", "Timing", "Verdict", "Detector", "Source", "CalibrationWarning",
]

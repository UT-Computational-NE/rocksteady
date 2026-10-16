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
   window keeps the last decided state.
5. **Overrides**: domain rules that can force a label (for example, a
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

Everything but STEADY (and HOLDING, which is steady awaiting confirmation)
is "transient" in the two-class sense. The detector is causal; batch
evaluation of a record gives exactly the online labels.

This module holds the contracts only and depends on numpy alone. Concrete
criteria, disturbances, overrides and signal builders live in their own
modules and are registered by name in ``rocksteady.registry``.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Mapping, Optional, Sequence, Union

import numpy as np


class Label(str, Enum):
    DISTURBED = "disturbed"
    SETTLING = "settling"
    INDETERMINATE = "indeterminate"
    HOLDING = "holding"
    STEADY = "steady"

    def __str__(self) -> str:
        return self.value


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
    """

    name: str
    statistic: float
    uncertainty: float
    limit: float
    k: float

    @property
    def upper(self) -> float:
        return self.statistic + self.k * self.uncertainty

    @property
    def lower(self) -> float:
        return self.statistic - self.k * self.uncertainty

    @property
    def ratio(self) -> float:
        """
        Margin ratio: below 1 passes with margin, above 1 is not demonstrated.

        Infinite when the statistic or its uncertainty is not finite, or the
        limit is not positive, so the worst ratio over several criteria does
        not depend on their order.
        """
        if not (np.isfinite(self.statistic) and np.isfinite(self.uncertainty)) or not self.limit > 0:
            return np.inf
        r = self.upper / self.limit
        return np.inf if np.isnan(r) else float(r)

    @property
    def outcome(self) -> Outcome:
        if not np.isfinite(self.statistic) or not np.isfinite(self.uncertainty):
            return Outcome.UNDECIDED
        if self.upper <= self.limit:
            return Outcome.PASS
        if self.lower > self.limit:
            return Outcome.FAIL
        return Outcome.UNDECIDED


class Criterion(ABC):
    """
    A test on one signal over the trailing window of block means.

    ``min_blocks`` is the fewest blocks the criterion can decide on; the
    detector refuses a window shorter than the largest ``min_blocks``.
    """

    name: str
    signal: str
    min_blocks: int = 3

    @abstractmethod
    def evaluate(self, t_blocks: np.ndarray, y_blocks: np.ndarray) -> CriterionResult:
        """``t_blocks`` are block centre times in seconds; ``y_blocks`` the block means."""


# ---------------------------------------------------------------------------
# Disturbances and overrides
# ---------------------------------------------------------------------------


class Disturbance(ABC):
    """Per-sample detector of external action. ``update`` returns True while active."""

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
    """A domain rule that may force a label. Return None to defer to the detector."""

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

    sample_period_s : spacing of the samples fed to ``update``.
    block_s         : block length at which criteria are evaluated.
    window_s        : trailing window length (a multiple of ``block_s``).
    settle_s        : quiet time required after a disturbance before a window
                      is evaluated, in addition to the window itself.
    hold_s          : continuous passing time before HOLDING becomes STEADY.
    hold_when_undecided : an undecided window keeps the last decided state
                      instead of returning INDETERMINATE.
    undecided_counts_toward_hold : when an undecided window keeps a passing
                      state, it also advances the hold clock. If False, the
                      hold clock pauses until a window passes again.
    max_gap_s       : longest run of missing data that is tolerated without
                      clearing the history. Missing data are non-finite or
                      unreadable samples, and the samples implied by a jump
                      in ``t`` of more than 1.5 sample periods. With 0 (the
                      default) any missing data resets the detector.
    min_block_coverage : when gaps are tolerated, the smallest fraction of
                      valid samples every block in the window must have for
                      the criteria to be evaluated; otherwise the window is
                      undecided.
    """

    sample_period_s: float = 1.0
    block_s: float = 30.0
    window_s: float = 300.0
    settle_s: float = 0.0
    hold_s: float = 0.0
    hold_when_undecided: bool = False
    undecided_counts_toward_hold: bool = True
    max_gap_s: float = 0.0
    min_block_coverage: float = 0.5

    def __post_init__(self):
        if not self.sample_period_s > 0:
            raise ValueError("sample_period_s must be positive")
        if self.settle_s < 0 or self.hold_s < 0 or self.max_gap_s < 0:
            raise ValueError("settle_s, hold_s and max_gap_s must be non-negative")
        if not 0 < self.min_block_coverage <= 1:
            raise ValueError("min_block_coverage must be in (0, 1]")

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

    Times passed to ``update`` must increase. A step of more than 1.5
    sample periods is a gap; see ``Timing.max_gap_s``.
    """

    def __init__(
        self,
        signals: Sequence[Signal],
        criteria: Sequence[Criterion],
        disturbances: Sequence[Disturbance] = (),
        overrides: Sequence[Override] = (),
        timing: Timing = Timing(),
    ):
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
        names = {s.name for s in self.signals}
        for c in self.criteria:
            if c.signal not in names:
                raise ValueError(f"criterion {c.name!r} refers to unknown signal {c.signal!r}")
        need = max([3] + [getattr(c, "min_blocks", 3) for c in self.criteria])
        if timing.n_blocks < need:
            raise ValueError(f"window_s must span at least {need} blocks for these criteria "
                             f"(it spans {timing.n_blocks})")
        self._index = {s.name: i for i, s in enumerate(self.signals)}
        self._n = timing.window_samples
        self._b = timing.block_samples
        self._nb = timing.n_blocks
        self._t_blocks = (np.arange(self._nb) - (self._nb - 1) / 2.0) * timing.block_s
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

    def _blocks(self) -> Optional[np.ndarray]:
        """
        Block means over the window, shape (n_blocks, n_signals), or None if
        some block has too few valid samples (``Timing.min_block_coverage``).
        """
        w = self._window()[self._n - self._nb * self._b :].reshape(self._nb, self._b, -1)
        valid = np.isfinite(w[:, :, 0])  # missing samples are whole NaN rows
        if valid.all():
            return w.mean(axis=1)
        if (valid.mean(axis=1) < self.timing.min_block_coverage).any():
            return None
        return np.nanmean(w, axis=1)

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
        if self._last_t is not None:
            step = t - self._last_t
            if not step > 0:
                raise ValueError(f"times must increase: t={t} follows t={self._last_t}")
            if step > 1.5 * dt:
                self._tolerate_gap(int(round(step / dt)) - 1)

        try:
            values = {s.name: s.value(sample) for s in self.signals}
            row = np.fromiter(values.values(), dtype=float, count=len(values))
            missing = not np.all(np.isfinite(row))
        except (KeyError, TypeError, ValueError):
            missing = True
        if missing:
            tolerated = self._tolerate_gap(1)
            self._last_t = t
            since_s = self._since_disturbance * dt if tolerated else 0.0
            return Verdict(t, Label.INDETERMINATE, since_disturbance_s=since_s)
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

        # Overrides observe every sample (they may keep state), but a
        # disturbance takes precedence over any label they force.
        ctx = Context(t, sample, values, disturbed, onset, since_s)
        forced = [(o.name, o.apply(ctx)) for o in self.overrides]

        if disturbed:
            self._record(False)
            return Verdict(t, Label.DISTURBED, since_disturbance_s=0.0)

        for name, label in forced:
            if label is not None:
                passing = label in (Label.STEADY, Label.HOLDING)
                self._record(passing if label != Label.INDETERMINATE else None)
                if passing:
                    label = self._passing_label()
                return Verdict(t, label, since_disturbance_s=since_s, forced_by=name)

        if self._count < self._n:
            return Verdict(t, Label.INDETERMINATE, since_disturbance_s=since_s)
        if self._since_disturbance < self.timing.quiet_samples:
            if not self._seen_disturbance:
                # Start of a record: the settling wait applies, but nothing was disturbed.
                self._record(None)
                return Verdict(t, Label.INDETERMINATE, since_disturbance_s=since_s)
            self._record(False)
            return Verdict(t, Label.SETTLING, since_disturbance_s=since_s)

        blocks = self._blocks()
        results = {}
        if blocks is None:
            outcomes = [Outcome.UNDECIDED]
        else:
            evaluated = [c.evaluate(self._t_blocks, blocks[:, self._index[c.signal]]) for c in self.criteria]
            outcomes = [r.outcome for r in evaluated]  # every criterion counts, whatever its name
            results = {c.name: r for c, r in zip(self.criteria, evaluated)}

        if Outcome.FAIL in outcomes:
            self._record(False)
            return Verdict(t, Label.SETTLING, results, since_s)
        if all(o == Outcome.PASS for o in outcomes):
            self._record(True)
            return Verdict(t, self._passing_label(), results, since_s)
        # Undecided.
        if self.timing.hold_when_undecided and self._last_decided is not None:
            if not (self._last_decided and not self.timing.undecided_counts_toward_hold):
                self._record(self._last_decided)
            return Verdict(t, self._passing_label() if self._last_decided else Label.SETTLING, results, since_s)
        self._record(None)
        return Verdict(t, Label.INDETERMINATE, results, since_s)


__all__ = [
    "Label", "Outcome", "Signal", "Criterion", "CriterionResult", "Disturbance", "Override",
    "Context", "Timing", "Verdict", "Detector", "Source",
]

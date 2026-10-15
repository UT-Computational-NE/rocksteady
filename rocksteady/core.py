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
   reactor below its power floor is steady by definition).

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
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Mapping, Optional, Sequence, Union

import numpy as np

from .stats import linear_fit


class Label(str, Enum):
    DISTURBED = "disturbed"
    SETTLING = "settling"
    INDETERMINATE = "indeterminate"
    HOLDING = "holding"
    STEADY = "steady"

    def __str__(self) -> str:
        return self.value


class Test(str, Enum):
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
        """Margin ratio: below 1 passes with margin, above 1 is not demonstrated."""
        return self.upper / self.limit if self.limit > 0 else np.inf

    @property
    def test(self) -> Test:
        if not np.isfinite(self.statistic) or not np.isfinite(self.uncertainty):
            return Test.UNDECIDED
        if self.upper <= self.limit:
            return Test.PASS
        if self.lower > self.limit:
            return Test.FAIL
        return Test.UNDECIDED


class Criterion(ABC):
    """A test on one signal over the trailing window of block means."""

    name: str
    signal: str

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
    """

    sample_period_s: float = 1.0
    block_s: float = 30.0
    window_s: float = 300.0
    settle_s: float = 0.0
    hold_s: float = 0.0
    hold_when_undecided: bool = False

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
        """Worst margin ratio over the criteria (the loop plan's eta_drift)."""
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
        names = {s.name for s in self.signals}
        for c in self.criteria:
            if c.signal not in names:
                raise ValueError(f"criterion {c.name!r} refers to unknown signal {c.signal!r}")
        if timing.n_blocks < 3:
            raise ValueError("window_s must span at least 3 blocks")
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
        self._last_decided: Optional[bool] = None  # True = passing
        self._pass_run = 0
        for d in self.disturbances:
            d.reset()
        for o in self.overrides:
            o.reset()

    # -- helpers ----------------------------------------------------------

    def _window(self) -> np.ndarray:
        """Window samples in time order, shape (n, n_signals)."""
        return np.concatenate((self._buf[self._head :], self._buf[: self._head]))

    def _blocks(self) -> np.ndarray:
        """Block means over the window, shape (n_blocks, n_signals)."""
        w = self._window()[self._n - self._nb * self._b :]
        return w.reshape(self._nb, self._b, -1).mean(axis=1)

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
        try:
            values = {s.name: s.value(sample) for s in self.signals}
        except (KeyError, TypeError, ValueError):
            self.reset()
            return Verdict(t, Label.INDETERMINATE)
        row = np.fromiter(values.values(), dtype=float, count=len(values))
        if not np.all(np.isfinite(row)):
            self.reset()
            return Verdict(t, Label.INDETERMINATE)

        self._buf[self._head] = row
        self._head = (self._head + 1) % self._n
        self._count += 1

        # Update every detector (no short-circuit) so their histories stay current.
        flags = [d.update(t, sample, values) for d in self.disturbances]
        disturbed = any(flags)
        onset = disturbed and not self._was_disturbed
        self._was_disturbed = disturbed
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
            self._record(False)
            return Verdict(t, Label.SETTLING, since_disturbance_s=since_s)

        blocks = self._blocks()
        results = {}
        for c in self.criteria:
            results[c.name] = c.evaluate(self._t_blocks, blocks[:, self._index[c.signal]])
        tests = [r.test for r in results.values()]

        if Test.FAIL in tests:
            self._record(False)
            return Verdict(t, Label.SETTLING, results, since_s)
        if all(tt == Test.PASS for tt in tests):
            self._record(True)
            return Verdict(t, self._passing_label(), results, since_s)
        # Undecided.
        if self.timing.hold_when_undecided and self._last_decided is not None:
            self._record(self._last_decided)
            return Verdict(t, self._passing_label() if self._last_decided else Label.SETTLING, results, since_s)
        self._record(None)
        self._pass_run = 0
        return Verdict(t, Label.INDETERMINATE, results, since_s)


def linear_residuals(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Residuals of ``y`` about its OLS line on ``t`` (re-exported for criteria)."""
    return linear_fit(t, y)[2]


__all__ = [
    "Label", "Test", "Signal", "Criterion", "CriterionResult", "Disturbance", "Override",
    "Context", "Timing", "Verdict", "Detector", "linear_residuals",
]

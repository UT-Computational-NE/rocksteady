"""
UT TRIGA steady-state detector expressed with the ``rocksteady`` library.

This file belongs in ``triga_modsim_tools/dataproc/``. It holds everything
reactor-specific: the signals, the calibrated tolerances and noise model,
the rod-move detectors, the zero-power rule, and a wrapper that keeps the
``OnlineSteadyStateTagger.update(power, rods) -> str`` interface that
``data_tagger.py`` already uses.

Definition (unchanged from steady_state_classifier.py)
------------------------------------------------------
Over the trailing 300 s window, on 30 s block means:
- true power stays within 2.5% of its window mean;
- each true rod trajectory stays within a tolerance of a straight line.
Power below 1 W is zero. A downward crossing of 1 W makes the reactor steady
at once (zero-power rule). An undecided window keeps the last decision.
"""

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np

from rocksteady import (
    Context,
    DeviationCriterion,
    Detector,
    Label,
    Override,
    Signal,
    StepChange,
    Timing,
)

DEFAULT_RODS = ("Tran", "Shim1", "Shim2", "Reg")

# Legacy label strings used by data_tagger.py.
STEADY, ACTIVE, SETTLING, INDETERMINATE = "steady", "active", "settling", "indeterminate"
LEGACY = {Label.STEADY: STEADY, Label.HOLDING: STEADY, Label.DISTURBED: ACTIVE,
          Label.SETTLING: SETTLING, Label.INDETERMINATE: INDETERMINATE}


@dataclass(frozen=True)
class TrigaCriteria:
    """Calibrated values (see calibrate_noise.py and the archive calibration notes)."""

    window_s: float = 300.0
    block_s: float = 30.0
    sample_period_s: float = 1.0
    power_tol: float = 0.025
    power_noise_by_decade: Mapping[int, float] = field(
        default_factory=lambda: {0: 0.0015, 1: 0.0010, 2: 0.0030, 3: 0.0026, 4: 0.0030, 5: 0.0031, 6: 0.0031})
    rod_tol: Mapping[str, float] = field(default_factory=lambda: {"Tran": 2.0, "Shim1": 1.5, "Shim2": 1.5, "Reg": 5.5})
    rod_noise: Mapping[str, float] = field(default_factory=lambda: {"Tran": 0.17, "Shim1": 0.10, "Shim2": 0.09, "Reg": 0.10})
    k_sigma: float = 3.0
    power_zero_w: float = 1.0
    zero_release_factor: float = 2.0
    rod_move_window_s: float = 10.0
    rod_move_tol: Mapping[str, float] = field(default_factory=lambda: {"Tran": 2.5, "Shim1": 2.5, "Shim2": 2.5, "Reg": 8.0})
    hold_when_undecided: bool = True

    def power_noise(self, level: float) -> float:
        """Relative noise of a power block mean; zeroed power has none."""
        if level <= 0:
            return 0.0
        keys = np.array(sorted(self.power_noise_by_decade))
        dec = int(np.floor(np.log10(level)))
        return float(self.power_noise_by_decade[int(keys[np.argmin(np.abs(keys - dec))])])


class ZeroPowerRule(Override):
    """
    Below the power floor the reactor is steady by definition.

    Latches on a downward crossing of the floor; releases on a new
    disturbance or when power rises above ``release_factor * floor``.
    """

    name = "zero_power"

    def __init__(self, floor_w: float, release_factor: float, power_signal: str = "power"):
        self.floor, self.release, self.signal = floor_w, release_factor, power_signal
        self.reset()

    def reset(self) -> None:
        self._prev = np.nan
        self._latched = False

    def apply(self, ctx: Context) -> Optional[Label]:
        p = ctx.values[self.signal]
        if p == 0.0 and self._prev > 0.0:
            self._latched = True
        elif p > self.release * self.floor or ctx.disturbance_onset:
            self._latched = False
        self._prev = p
        return Label.STEADY if self._latched else None


def build_detector(c: TrigaCriteria = TrigaCriteria(), rods: Sequence[str] = DEFAULT_RODS,
                   power_key: str = "LinPower") -> Detector:
    """Assemble the reactor detector from library parts."""
    zero = c.power_zero_w
    signals = [Signal("power", lambda s, k=power_key: 0.0 if s[k] < zero else float(s[k]))]
    signals += [Signal(r, r) for r in rods]
    criteria = [DeviationCriterion("power", c.power_tol, c.power_noise, model="constant",
                                   relative=True, scale_floor=zero, k=c.k_sigma, name="power")]
    criteria += [DeviationCriterion(r, c.rod_tol[r], c.rod_noise[r], model="linear", k=c.k_sigma, name=r)
                 for r in rods]
    m = max(1, int(round(c.rod_move_window_s / c.sample_period_s)))
    disturbances = [StepChange(r, c.rod_move_tol[r], m) for r in rods]
    overrides = [ZeroPowerRule(zero, c.zero_release_factor)]
    timing = Timing(sample_period_s=c.sample_period_s, block_s=c.block_s, window_s=c.window_s,
                    settle_s=0.0, hold_s=0.0, hold_when_undecided=c.hold_when_undecided)
    return Detector(signals, criteria, disturbances, overrides, timing)


class OnlineSteadyStateTagger:
    """
    Drop-in replacement for the previous class of the same name.

    Feed 1 s block averages (power mean, rod median) and get the legacy
    label string. ``last_verdict`` holds the full library verdict.
    """

    def __init__(self, criteria: Optional[TrigaCriteria] = None, rods: Sequence[str] = DEFAULT_RODS):
        self.criteria = criteria or TrigaCriteria()
        self.rods = tuple(rods)
        self.detector = build_detector(self.criteria, self.rods)
        self._t = 0.0
        self.last_verdict = None

    def reset(self) -> None:
        self.detector.reset()

    def update(self, power: float, rod_positions: Sequence[float]) -> str:
        sample = {"LinPower": power, **dict(zip(self.rods, rod_positions))}
        self.last_verdict = self.detector.update(self._t, sample)
        self._t += self.criteria.sample_period_s
        return LEGACY[self.last_verdict.label]

    @property
    def last_diagnostics(self) -> dict:
        v = self.last_verdict
        return {} if v is None else {f"{n}_dev": r.statistic for n, r in v.results.items()}

"""
STNCL natural-circulation loop steady-state detector expressed with the
``rocksteady`` library.

This file belongs in the STNCL analysis repository. It holds everything
loop-specific: the channel groups and derived signals, the acceptance
classes (drift limits, evaluation window, hold), the settling precondition,
and the logged-event disturbances. The drift equivalence test itself
(``|b| + 2 u_b <= b_crit`` with a Newey-West ``u_b``) is the library's
``DriftCriterion``.

Definition (Phase 1 plan, Section 2.1.4)
----------------------------------------
Over the trailing evaluation window, on 30 s block means, every mandatory
signal (16 thermocouples and 5 derived quantities) has no resolvable drift.
Evaluation does not begin until ``tau_slow_op`` has elapsed since the last
logged control or configuration event. The first passing window starts the
hold clock; all criteria must keep passing for the class hold time before
the state is accepted. Any later event resets the sequence.
"""

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence, Tuple

import numpy as np

from rocksteady import Detector, DriftCriterion, EventLog, Signal, Timing

FLUID = ("PTC1", "PTC2", "PTC3", "PTC4")
WALL = tuple(f"STC{i}" for i in range(1, 13))


@dataclass(frozen=True)
class AcceptanceClass:
    """Drift limits in K/min; times in seconds (Phase 1 plan, Table 2)."""

    name: str
    b_fluid: float
    b_wall: float
    b_diag: float
    window_s: float
    hold_s: float
    report_s: float


CLASSES = {
    "operating": AcceptanceClass("operating", 0.010, 0.020, 0.005, 1200.0, 1200.0, 1200.0),
    "calibration": AcceptanceClass("calibration", 0.005, 0.010, 0.0025, 2700.0, 2700.0, 1800.0),
}


def _mean(cols: Sequence[str]):
    return lambda s: float(np.mean([s[c] for c in cols]))


def _spread(cols: Sequence[str]):
    return lambda s: float(max(s[c] for c in cols) - min(s[c] for c in cols))


def loop_signals(fluid: Sequence[str] = FLUID, wall: Sequence[str] = WALL) -> Tuple[Sequence[Signal], Mapping[str, str]]:
    """Mandatory signals and their group ('fluid', 'wall', 'diag')."""
    signals, group = [], {}
    for c in fluid:
        signals.append(Signal(c, c)); group[c] = "fluid"
    for c in wall:
        signals.append(Signal(c, c)); group[c] = "wall"
    derived = [
        ("T_avg", _mean(fluid), "fluid"),
        ("fluid_spread", _spread(fluid), "fluid"),
        ("wall_spread", _spread(wall), "wall"),
        ("intwall_offset", lambda s: _mean(fluid)(s) - _mean(wall)(s), "wall"),
        ("T_diag", _mean(tuple(fluid) + tuple(wall)), "diag"),
    ]
    for name, src, g in derived:
        signals.append(Signal(name, src)); group[name] = g
    return signals, group


def build_detector(class_name: str = "operating", tau_slow_op_s: float = 0.0,
                   events: Iterable[Tuple[float, float]] = (), sample_period_s: float = 1.0,
                   block_s: float = 30.0, k: float = 2.0, hold: bool = True) -> Detector:
    """
    Assemble the loop detector for an acceptance class.

    Parameters
    ----------
    class_name : "operating" or "calibration"
    tau_slow_op_s : float
        Settling precondition after a logged event (from Phase 2; 0 until
        it is measured).
    events : iterable of (start_s, duration_s)
        Logged control, configuration or alarm events for the run.
    hold : bool
        Apply the class hold time (False gives the bare window test, which
        is what the post-run plotting of eta_drift uses).
    """
    cls = CLASSES[class_name]
    signals, group = loop_signals()
    limit = {"fluid": cls.b_fluid, "wall": cls.b_wall, "diag": cls.b_diag}
    criteria = [DriftCriterion(s.name, limit[group[s.name]] / 60.0, k=k, name=s.name) for s in signals]
    timing = Timing(sample_period_s=sample_period_s, block_s=block_s, window_s=cls.window_s,
                    settle_s=tau_slow_op_s, hold_s=cls.hold_s if hold else 0.0, hold_when_undecided=False)
    return Detector(signals, criteria, [EventLog(events)], overrides=(), timing=timing)

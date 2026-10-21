"""
Disturbance detectors: external action that resets the detector's clocks.

- ``StepChange``: a signal's mean over the latest ``m`` samples differs from
  its mean over the ``m`` before by more than a threshold. Catches discrete
  actuator moves (valve or positioner steps) while rejecting jitter and servo
  hunting.
- ``EventLog``: externally logged events (control, configuration, alarm)
  with a start time and a duration: any control, configuration or alarm
  event resets the sequence.
"""

from collections import deque
from typing import Deque, Iterable, Mapping, Optional, Sequence, Tuple, cast

import numpy as np

from . import _validate as _v
from . import registry
from .core import Disturbance


@registry.disturbances.register("step")
class StepChange(Disturbance):
    """
    Discrete move detector on one signal.

    Parameters
    ----------
    signal : str
        Signal name (a key of the detector's computed values).
    threshold : float
        Change in the ``m``-sample mean that counts as a move.
    m : int
        Averaging length in samples.
    """

    def __init__(self, signal: str, threshold: float, m: int = 10, name: Optional[str] = None):
        what = f"StepChange {name or signal!r}"
        threshold = _v.finite(threshold, f"{what}: threshold", positive=True)
        m = _v.positive_int(m, f"{what}: m")
        self.signal, self.threshold, self.m = signal, threshold, m
        self.name = name or f"{signal}_step"
        self.reset()

    def reset(self) -> None:
        self._hist: Deque[float] = deque(maxlen=2 * self.m)

    def update(self, t: float, sample: Mapping[str, float], values: Mapping[str, float]) -> bool:
        self._hist.append(values[self.signal])
        if len(self._hist) < 2 * self.m:
            return False
        h = np.fromiter(self._hist, dtype=float, count=2 * self.m)
        return bool(abs(h[self.m :].mean() - h[: self.m].mean()) > self.threshold)


@registry.disturbances.register("event_log")
class EventLog(Disturbance):
    """
    Logged events as disturbances.

    Parameters
    ----------
    events : iterable of (start_s, duration_s)
        Each event is active over ``[start, start + duration]``. It marks
        every sample whose interval ``(t_prev, t]`` it overlaps, so a
        duration of 0 marks an instant and flags exactly one sample. The
        first sample after construction or ``reset`` has no previous time:
        it is flagged by any event whose closed span contains it, including
        an instantaneous event at that time and an event already under way.
        Events that ended before the first sample are not replayed.
    """

    def __init__(self, events: Iterable[Tuple[float, float]] = (), name: str = "event_log"):
        self.name = name
        checked = []
        for i, ev in enumerate(events):
            try:
                start, dur = ev
            except (TypeError, ValueError):
                raise ValueError(f"EventLog {name!r}: event {i} must be (start_s, duration_s), got {ev!r}") from None
            checked.append((_v.finite(start, f"EventLog {name!r}: event {i} start"),
                            _v.finite(dur, f"EventLog {name!r}: event {i} duration", nonnegative=True)))
        self.events: Sequence[Tuple[float, float]] = sorted(checked)
        self.reset()

    def reset(self) -> None:
        self._i = 0  # first event that may still be active
        self._last_t: Optional[float] = None

    def update(self, t: float, sample: Mapping[str, float], values: Mapping[str, float]) -> bool:
        first = self._last_t is None
        lo = t if first else cast(float, self._last_t)  # samples cover [t, t] at first, then (t_prev, t]
        self._last_t = t

        def ended(end: float) -> bool:
            return end < lo if first else end <= lo

        events = self.events
        while self._i < len(events) and ended(events[self._i][0] + events[self._i][1]):
            self._i += 1
        for k in range(self._i, len(events)):  # sorted by start: stop at the first event not yet begun
            start, dur = events[k]
            if start > t:
                break
            if not ended(start + dur):
                return True
        return False

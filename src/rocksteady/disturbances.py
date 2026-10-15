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
from typing import Iterable, Mapping, Sequence, Tuple

import numpy as np

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

    def __init__(self, signal: str, threshold: float, m: int = 10, name: str = None):
        self.signal, self.threshold, self.m = signal, threshold, m
        self.name = name or f"{signal}_step"
        self.reset()

    def reset(self) -> None:
        self._hist = deque(maxlen=2 * self.m)

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
        Each event is active over ``[start, start + duration]`` and marks
        every sample whose interval ``(t - dt, t]`` it overlaps. A duration
        of 0 marks an instant and flags exactly one sample.
    """

    def __init__(self, events: Iterable[Tuple[float, float]] = (), name: str = "event_log"):
        self.name = name
        self.events: Sequence[Tuple[float, float]] = sorted((float(s), float(d)) for s, d in events)
        self.reset()

    def reset(self) -> None:
        self._i = 0  # first event that may still be active
        self._last_t = None

    def update(self, t: float, sample: Mapping[str, float], values: Mapping[str, float]) -> bool:
        dt = 0.0 if self._last_t is None else t - self._last_t
        self._last_t = t
        # An event is active for the sample at t if it overlaps (t - dt, t].
        while self._i < len(self.events) and self.events[self._i][0] + self.events[self._i][1] <= t - dt:
            self._i += 1
        for start, dur in self.events[self._i :]:
            if start > t:
                break
            if start + dur > t - dt:
                return True
        return False

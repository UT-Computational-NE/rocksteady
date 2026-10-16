"""
Overrides: domain rules that may force a label.

- ``FloorLatch``: once a signal falls below a floor, the system is in a
  known state (for example, shut down) and the label is forced, until the
  signal rises well above the floor or a new disturbance begins.
"""

from typing import Optional

import numpy as np

from . import registry
from .core import Context, Label, Override


@registry.overrides.register("floor_latch")
class FloorLatch(Override):
    """
    Force a label after a downward crossing of a floor.

    Latches when the signal goes from at or above ``floor`` to below it.
    Releases when the signal exceeds ``release_factor * floor`` or when a
    disturbance begins; a disturbance beginning takes priority, so a crossing
    on the same sample does not latch. A later downward crossing, including
    one while a disturbance is still active, latches again (the detector
    reports DISTURBED until the disturbance ends). While latched the label
    is ``label`` (STEADY by default: a system below its operating floor is
    steady by definition).

    Parameters
    ----------
    signal : str
        Signal name (a key of the detector's computed values).
    floor : float
    release_factor : float
        Release threshold as a multiple of ``floor`` (hysteresis).
    label : Label or its string value
    name : str
    """

    def __init__(self, signal: str, floor: float, release_factor: float = 2.0,
                 label: Label = Label.STEADY, name: str = "floor_latch"):
        self.signal, self.floor, self.release_factor = signal, floor, release_factor
        self.label, self.name = Label(label), name
        self.reset()

    def reset(self) -> None:
        self._prev = np.nan
        self._latched = False

    def apply(self, ctx: Context) -> Optional[Label]:
        v = ctx.values[self.signal]
        if ctx.disturbance_onset or v > self.release_factor * self.floor:
            # A disturbance beginning releases the latch, even on the sample where the signal crosses the floor.
            self._latched = False
        elif v < self.floor and self._prev >= self.floor:
            self._latched = True
        self._prev = v
        return self.label if self._latched else None


__all__ = ["FloorLatch"]

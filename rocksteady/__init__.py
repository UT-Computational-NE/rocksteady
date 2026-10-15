"""
rocksteady: a reusable pattern for steady-state detection on experiments
with characteristic time constants, several signals, and external actions.

See the README for the design and ``examples/`` for worked detectors.
"""

from .core import (
    Context,
    Criterion,
    CriterionResult,
    Detector,
    Disturbance,
    Label,
    Override,
    Signal,
    Test,
    Timing,
    Verdict,
)
from .criteria import BoundCriterion, DeviationCriterion, DriftCriterion
from .disturbances import EventLog, StepChange
from .overrides import FloorLatch
from . import batch, signals, stats

__version__ = "0.1.0"

__all__ = [
    "Context", "Criterion", "CriterionResult", "Detector", "Disturbance", "Label", "Override",
    "Signal", "Test", "Timing", "Verdict", "BoundCriterion", "DeviationCriterion", "DriftCriterion",
    "EventLog", "FloorLatch", "StepChange", "batch", "signals", "stats",
]

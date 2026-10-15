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
    Outcome,
    Override,
    Signal,
    Timing,
    Verdict,
)
from .criteria import BoundCriterion, DeviationCriterion, DriftCriterion
from .disturbances import EventLog, StepChange
from .overrides import FloorLatch
from .factory import build_detector, criterion, disturbance, override
from . import batch, factory, registry, signals, stats

__version__ = "0.2.0"

__all__ = [
    "Context", "Criterion", "CriterionResult", "Detector", "Disturbance", "Label", "Override",
    "Outcome", "Signal", "Timing", "Verdict", "BoundCriterion", "DeviationCriterion", "DriftCriterion",
    "EventLog", "FloorLatch", "StepChange", "build_detector", "criterion", "disturbance", "override",
    "batch", "factory", "registry", "signals", "stats",
]

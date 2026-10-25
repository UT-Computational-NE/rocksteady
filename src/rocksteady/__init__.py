"""
rocksteady: a reusable pattern for steady-state detection on experiments
with characteristic time constants, several signals, and external actions.

See the README for the design and ``examples/`` for worked detectors.
"""

from .core import (
    CalibrationWarning,
    Context,
    Criterion,
    CriterionResult,
    Detector,
    Disturbance,
    Label,
    Outcome,
    Override,
    Reason,
    Signal,
    Timing,
    Verdict,
)
from .criteria import AsymptoteCriterion, BoundCriterion, DeviationCriterion, DriftCriterion
from .disturbances import EventLog, StepChange
from .overrides import FloorLatch
from .factory import build_detector, criterion, disturbance, override
from .states import Segment, State, segments, segments_from
from .optimal import Subtype, classify_segments, compare_segments, optimal_segments
from .timing import TimingSuggestion, suggest_timing
from . import batch, factory, fit, metrics, optimal, registry, report, signals, states, stats, timing

__version__ = "0.5.1"

__all__ = [
    "Context", "Criterion", "CriterionResult", "Detector", "Disturbance", "Label", "Override", "Reason",
    "Outcome", "Signal", "Timing", "Verdict", "AsymptoteCriterion", "BoundCriterion", "DeviationCriterion",
    "DriftCriterion",
    "EventLog", "FloorLatch", "StepChange", "build_detector", "criterion", "disturbance", "override",
    "Segment", "State", "segments", "segments_from", "optimal_segments", "classify_segments", "compare_segments",
    "Subtype", "suggest_timing", "TimingSuggestion", "CalibrationWarning",
    "batch", "factory", "fit", "metrics", "optimal", "registry", "timing", "report", "signals", "states", "stats",
]

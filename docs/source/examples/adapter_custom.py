"""
Custom components for an adapter: a criterion, a disturbance, an override
and a slope method, registered by name so that plain-data specs can use them.

The physics is illustrative: a level held by a valve, a command channel the
operator moves, and a maintenance flag. Run it with
``python docs/source/examples/adapter_custom.py``.
"""

from typing import Mapping, Optional

import numpy as np
import pandas as pd

import rocksteady as rs
from rocksteady import Context, Criterion, CriterionResult, Disturbance, Label, Override, registry


# [criterion]
@registry.criteria.register("range")
class RangeCriterion(Criterion):
    """Peak-to-peak of the block means within a limit; uncertainty from a noise model."""

    min_blocks = 3

    def __init__(self, signal: str, limit: float, noise: float, k: float = 3.0, name: Optional[str] = None):
        self.signal, self.limit, self.noise, self.k = signal, limit, noise, k
        self.name = name or f"{signal}_range"

    def evaluate(self, t_blocks: np.ndarray, y_blocks: np.ndarray) -> CriterionResult:
        # statistic, its standard uncertainty, the limit, and the coverage factor k:
        # PASS if statistic + k u <= limit, FAIL if statistic - k u > limit, else UNDECIDED.
        return CriterionResult(self.name, float(np.ptp(y_blocks)), self.noise, self.limit, self.k)
# [criterion]


# [disturbance]
@registry.disturbances.register("command_change")
class CommandChange(Disturbance):
    """Active while a command channel differs from its previous sample by more than ``tol``."""

    def __init__(self, signal: str, tol: float, name: Optional[str] = None):
        self.signal, self.tol = signal, tol
        self.name = name or f"{signal}_change"
        self.reset()

    def reset(self) -> None:  # called by Detector.reset() and after a gap that clears history
        self._prev = None

    def update(self, t: float, sample: Mapping[str, float], values: Mapping[str, float]) -> bool:
        v = values[self.signal]  # computed signal values, by name
        moved = self._prev is not None and abs(v - self._prev) > self.tol
        self._prev = v
        return moved
# [disturbance]


# [override]
@registry.overrides.register("maintenance")
class Maintenance(Override):
    """Force INDETERMINATE while a maintenance flag column is set; defer otherwise."""

    def __init__(self, column: str = "maintenance", name: str = "maintenance"):
        self.column, self.name = column, name

    def apply(self, ctx: Context) -> Optional[Label]:
        return Label.INDETERMINATE if ctx.sample.get(self.column, 0) else None
# [override]


# [slope]
def slope_iid(t, y, k: float = 2.0):
    """Ordinary least-squares slope with the classical (independent errors) standard error."""
    tc = t - t.mean()
    stt = float(tc @ tc)
    b = float(tc @ (y - y.mean())) / stt
    e = y - y.mean() - b * tc
    u = float(np.sqrt((e @ e) / (len(y) - 2) / stt))
    return b, u, k


def _check_iid(k: float = 2.0) -> None:  # checked when a DriftCriterion is built, before any data
    if not k >= 0:
        raise ValueError(f"k must be >= 0, got {k!r}")


slope_iid.validate_options = _check_iid
registry.slope_methods.register("iid", slope_iid)
# [slope]


# [spec]
SPEC = {
    "timing": {"sample_period_s": 1.0, "block_s": 10.0, "window_s": 120.0, "settle_s": 30.0},
    "signals": ["level", "command"],
    "criteria": [
        {"type": "range", "signal": "level", "limit": 0.5, "noise": 0.02},
        {"type": "drift", "signal": "level", "limit_per_s": 2e-3, "method": "iid", "method_options": {"k": 3.0}},
    ],
    "disturbances": [{"type": "command_change", "signal": "command", "tol": 0.5}],
    "overrides": [{"type": "maintenance"}],
}
# [spec]


def synthetic_record(n: int = 1200, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    command = np.where(t < 600, 40.0, 55.0)
    level = np.empty(n)
    level[0] = 40.0
    for i in range(1, n):
        level[i] = level[i - 1] + (command[i] - level[i - 1]) / 20.0
    maintenance = ((t >= 900) & (t < 960)).astype(int)
    return pd.DataFrame({"level": level + rng.normal(0, 0.05, n), "command": command, "maintenance": maintenance})


if __name__ == "__main__":
    det = rs.build_detector(SPEC)
    df = synthetic_record()
    out = rs.batch.run(det, df, with_results=True)
    print(out.groupby(["label", "reason"]).size().to_string())
    assert out["label"][600] == "disturbed"                                # the command moved
    assert set(out["forced_by"][900:960]) == {"maintenance"}              # the flag forced the label
    assert "steady" in set(out["label"][:600]) and "steady" in set(out["label"][960:])
    assert {"ratio_level_range", "ratio_level_drift"} <= set(out.columns)

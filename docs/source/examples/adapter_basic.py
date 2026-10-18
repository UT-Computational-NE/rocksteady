"""
A minimal adapter: a heated loop with three thermocouples and an operator log.

The four parts every adapter has:

1. the calibrated values, in one frozen configuration;
2. the detector, built from a plain-data spec;
3. a streaming wrapper for live data;
4. a batch path for stored records: labels, segments and a fit report.

The facility, channel names and limits are illustrative. Run it with
``python docs/source/examples/adapter_basic.py``.
"""

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import pandas as pd

import rocksteady as rs


# [config]
@dataclass(frozen=True)
class LoopCriteria:
    """Everything the facility calibrates, in one place."""

    sample_period_s: float = 1.0
    block_s: float = 30.0
    window_s: float = 1200.0          # 40 blocks: enough for the "ar1" drift test
    settle_s: float = 300.0           # quiet time after a logged event
    hold_s: float = 120.0             # passing time before HOLDING becomes STEADY
    drift_limit_per_s: float = 2e-4   # K/s, on the loop mean
    spread_limit: float = 0.6         # K, largest allowed channel-to-channel spread
    channels: Tuple[str, ...] = ("T1", "T2", "T3")
# [config]


# [spec]
def detector_spec(c: LoopCriteria = LoopCriteria(), events: Sequence[Tuple[float, float]] = ()) -> dict:
    """The detector as plain data (JSON, YAML or TOML compatible)."""
    chans = list(c.channels)
    return {
        "timing": {"sample_period_s": c.sample_period_s, "block_s": c.block_s, "window_s": c.window_s,
                   "settle_s": c.settle_s, "hold_s": c.hold_s},
        "signals": [
            *chans,                                                  # raw columns
            {"name": "T_avg", "builder": "mean", "cols": chans},     # derived signals
            {"name": "T_spread", "builder": "spread", "cols": chans},
        ],
        "criteria": [
            {"type": "drift", "signal": "T_avg", "limit_per_s": c.drift_limit_per_s},
            {"type": "bound", "signal": "T_spread", "low": 0.0, "high": c.spread_limit},
        ],
        # Operator-log events (start_s, duration_s) reset the sequence: settle, then evaluate.
        "disturbances": [{"type": "event_log", "events": [list(e) for e in events]}],
    }


def build_detector(c: LoopCriteria = LoopCriteria(), events=()) -> rs.Detector:
    return rs.build_detector(detector_spec(c, events))
# [spec]


# [stream]
class LiveMonitor:
    """Feed one sample per period; get the label and the worst margin ratio."""

    def __init__(self, c: LoopCriteria = LoopCriteria(), events=()):
        self.detector = build_detector(c, events)

    def update(self, t: float, sample: dict) -> Tuple[str, float]:
        v = self.detector.update(t, sample)
        return v.label.value, v.eta
# [stream]


# [batch]
def qualify(df: pd.DataFrame, events=(), c: LoopCriteria = LoopCriteria()):
    """Labels, accepted intervals and a fit report for a stored record."""
    det = build_detector(c, events)
    out = rs.batch.run(det, df)               # one row per sample: label, reason, eta, ...
    segs = rs.segments_from(det, out)         # steady / transient / indeterminate, backfilled
    accepted = rs.batch.accepted_intervals(out["label"])
    report = rs.batch.fit_report(det, df, {"T_avg": {"steady": 0, "transient": "first_order"}},
                                 labels=out, units={"T_avg": "°C"})
    return out, segs, accepted, report
# [batch]


def synthetic_record(n: int = 7200, seed: int = 0):
    """Relaxation to 40 °C, a setpoint change to 45 °C at 3600 s, AR(1) noise per channel."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    target = np.where(t < 3600, 40.0, 45.0)
    base = np.empty(n)
    base[0] = 38.0
    for i in range(1, n):
        base[i] = base[i - 1] + (target[i] - base[i - 1]) / 300.0
    e = np.zeros((n, 3))
    for i in range(1, n):
        e[i] = 0.7 * e[i - 1] + rng.normal(0, 0.02, 3)
    idx = pd.date_range("2026-03-02 08:00", periods=n, freq="1s")
    df = pd.DataFrame(base[:, None] + e + [0.0, 0.2, -0.1], index=idx, columns=["T1", "T2", "T3"])
    return df, [(3600.0, 0.0)]


if __name__ == "__main__":
    df, events = synthetic_record()
    out, segs, accepted, report = qualify(df, events)
    print(out["label"].value_counts().to_string(), "\n")
    t = out["t_s"].to_numpy()  # positions, not the DatetimeIndex
    print("accepted intervals (s):", [(float(t[a]), float(t[b - 1])) for a, b in accepted])
    print(report.to_string(intervals=False))
    monitor = LiveMonitor(events=events)
    for ti, row in zip(t, df.to_dict("records")):
        label, eta = monitor.update(float(ti), row)
    print("\nlive, last sample:", label, f"eta={eta:.2f}")
    assert accepted, "the synthetic record settles twice"
    assert label == out["label"].iloc[-1]  # streaming and batch agree

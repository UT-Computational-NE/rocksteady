"""
Golden-label snapshot tests.

Two seeded synthetic records, one per detector pattern, are run through
``batch.run`` and compared with stored labels and margin ratios. A change
that alters labels fails here; if the change is intended, regenerate with

    PYTHONPATH=src python tests/test_regression.py

and record the reason in CHANGELOG.md.
"""

import io
import os

import numpy as np
import pandas as pd
import pytest

import rocksteady as rs
from rocksteady import Label

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class _FloorLatch(rs.Override):
    """Steady below a floor after a downward crossing (kept local so the snapshot tracks only the core)."""

    name = "floor"

    def __init__(self, signal, floor, release):
        self.signal, self.floor, self.release = signal, floor, release
        self.reset()

    def reset(self):
        self._prev, self._latched = np.nan, False

    def apply(self, ctx):
        v = ctx.values[self.signal]
        if v == 0.0 and self._prev > 0.0:
            self._latched = True
        elif v > self.release * self.floor or ctx.disturbance_onset:
            self._latched = False
        self._prev = v
        return Label.STEADY if self._latched else None


def actuator_case():
    """Level held by a stepping actuator, a level-dependent noise model, a floor rule, one dropout."""
    rng = np.random.default_rng(11)
    n = 2400
    act = np.zeros(n)
    act[400:] += 10.0
    act[1100:] += 15.0
    act[1700:] -= 25.0
    level = np.maximum(50.0 * act + 5.0, 0.0)
    for i in range(1, n):  # first-order response with a 30 s time constant
        level[i] = level[i - 1] + (max(50.0 * act[i] + 5.0, 0.0) - level[i - 1]) / 30.0
    level[1700:] *= np.exp(-np.arange(n - 1700) / 40.0)
    level = level * (1 + rng.normal(0, 0.004, n))
    act = act + rng.normal(0, 0.05, n)
    df = pd.DataFrame({"level": level, "act": act})
    df.loc[900, "level"] = np.nan

    floor = 1.0
    signals = [rs.Signal("level", lambda s: 0.0 if s["level"] < floor else float(s["level"])), rs.Signal("act", "act")]

    def noise(m):
        return 0.0 if m <= 0 else (0.003 if m < 100 else 0.002)

    criteria = [
        rs.DeviationCriterion("level", 0.025, noise, model="constant", relative=True, scale_floor=floor, name="level"),
        rs.DeviationCriterion("act", 1.0, 0.05, model="linear", name="act"),
    ]
    det = rs.Detector(signals, criteria, [rs.StepChange("act", 2.0, m=10)], [_FloorLatch("level", floor, 2.0)],
                      rs.Timing(sample_period_s=1.0, block_s=10.0, window_s=100.0, hold_when_undecided=True))
    return det, df


def drift_case():
    """Two channels relaxing to a setpoint with AR(1) noise, a logged setpoint change, settle and hold."""
    rng = np.random.default_rng(12)
    n = 3000
    t = np.arange(n, dtype=float)
    target = np.where(t < 1500, 40.0, 45.0)
    base = np.empty(n)
    base[0] = 30.0
    for i in range(1, n):
        base[i] = base[i - 1] + (target[i] - base[i - 1]) / 200.0
    e = np.zeros((n, 2))
    for i in range(1, n):
        e[i] = 0.7 * e[i - 1] + rng.normal(0, 0.02, 2)
    df = pd.DataFrame({"a": base + e[:, 0], "b": base + 0.5 + e[:, 1]})

    signals = [rs.Signal("a", "a"), rs.Signal("b", "b"),
               rs.Signal("mean", lambda s: 0.5 * (s["a"] + s["b"])),
               rs.Signal("spread", lambda s: abs(s["a"] - s["b"]))]
    # Pinned to the 0.1 slope method, so this snapshot guards that path; "ar1" is checked in test_calibration.
    criteria = [rs.DriftCriterion(name, 0.002 / 60.0 * 30, method="nw", name=name) for name in ("a", "b", "mean", "spread")]
    det = rs.Detector(signals, criteria, [rs.EventLog([(1500.0, 0.0), (2600.0, 20.0)])],
                      timing=rs.Timing(sample_period_s=1.0, block_s=10.0, window_s=300.0, settle_s=60.0, hold_s=120.0))
    return det, df


CASES = {"actuator": actuator_case, "drift": drift_case}


def _run(case):
    det, df = CASES[case]()
    out = rs.batch.run(det, df, with_results=True)
    cols = ["label", "eta", "limiting", "since_disturbance_s", "forced_by"]
    cols += sorted(c for c in out.columns if c.startswith("ratio_"))
    return out[cols]


def _read_golden(path):
    return pd.read_csv(path, keep_default_na=False, na_values=["nan", ""])


def _text(column):
    """The column as text, a missing entry as ``""``.

    A golden column with no text reads as float64; filling a float column with ``""`` is deprecated (Pandas4Warning
    from pandas 3.1), so the column becomes object first.
    """
    return column.astype(object).fillna("").astype(str).to_numpy()


def _assert_same_text(got, want):
    """``limiting`` and ``forced_by`` hold the same text, a missing entry counting as ``""``."""
    for col in ("limiting", "forced_by"):
        np.testing.assert_array_equal(_text(got[col]), _text(want[col]), err_msg=col)


@pytest.mark.parametrize("case", sorted(CASES))
def test_golden(case):
    got = _run(case)
    want = _read_golden(os.path.join(DATA, f"golden_{case}.csv"))
    assert list(got.columns) == list(want.columns)
    np.testing.assert_array_equal(got["label"].to_numpy(), want["label"].to_numpy())
    _assert_same_text(got, want)
    for col in [c for c in got.columns if c not in ("label", "limiting", "forced_by")]:
        np.testing.assert_allclose(got[col].to_numpy(float), want[col].to_numpy(float), rtol=1e-9, equal_nan=True)


def test_the_text_comparison_matches_missing_text_and_nothing_else():
    """A golden column with no text reads as float64 NaN; it equals a column of missing text, and nothing else."""
    want = _read_golden(io.StringIO("label,limiting,forced_by\nsteady,a,\nsteady,,\n"))
    assert want["forced_by"].dtype == float
    _assert_same_text(pd.DataFrame({"limiting": ["a", None], "forced_by": [None, None]}), want)
    differing = ((["a", "b"], [None, None]), (["a", None], ["floor", None]), (["a", None], [None, "floor"]))
    for limiting, forced_by in differing:
        with pytest.raises(AssertionError):
            _assert_same_text(pd.DataFrame({"limiting": limiting, "forced_by": forced_by}), want)


def test_golden_cases_exercise_every_label():
    for case in CASES:
        labels = set(_run(case)["label"])
        assert {"settling", "indeterminate", "steady"} <= labels
    assert "disturbed" in set(_run("actuator")["label"]) and "holding" in set(_run("drift")["label"])


if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    for name in CASES:
        out = _run(name)
        out.to_csv(os.path.join(DATA, f"golden_{name}.csv"), index=False, float_format="%.12g")
        print(name, out["label"].value_counts().to_dict())

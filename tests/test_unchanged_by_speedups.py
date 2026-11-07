"""
The cheaper derived properties, deviation criterion, line fit, step detector and run loop (#180) give 0.5's output
bit for bit. The 0.5 implementations they replaced are kept here as references, and the comparisons use ``==`` on
the bits of every float, so a change that moves only a last bit, or turns 0.0 into -0.0, fails.
"""

import math
import warnings

import numpy as np
import pandas as pd
import pytest

import rocksteady as rs
from rocksteady import Label
from rocksteady.batch import RUN_COLUMNS, _detail_columns, _unique_columns
from rocksteady.core import Outcome

# --- the 0.5 implementations ------------------------------------------------


class _Result0(rs.CriterionResult):
    """A ``CriterionResult`` whose derived properties are the 0.5 ones, recomputed on every read."""

    @property
    def upper(self):
        return self.statistic + self.k * self.uncertainty

    @property
    def lower(self):
        return self.statistic - self.k * self.uncertainty

    @property
    def invalid_reason(self):
        for label, value in (("statistic", self.statistic), ("uncertainty", self.uncertainty),
                             ("limit", self.limit), ("k", self.k)):
            if not np.isfinite(value):
                return f"{label} is not finite ({value!r})"
        if self.uncertainty < 0:
            return f"uncertainty is negative ({self.uncertainty!r})"
        if self.k < 0:
            return f"coverage factor k is negative ({self.k!r})"
        if not self.limit > 0:
            return f"limit is not positive ({self.limit!r})"
        return None

    @property
    def ratio(self):
        if self.invalid_reason is not None:
            return np.inf
        r = self.upper / self.limit
        return np.inf if np.isnan(r) else float(r)

    @property
    def outcome(self):
        if self.invalid_reason is not None:
            return Outcome.UNDECIDED
        if self.upper <= self.limit:
            return Outcome.PASS
        if self.lower > self.limit:
            return Outcome.FAIL
        return Outcome.UNDECIDED


def _linear_fit0(t, y):
    """The 0.5 ``stats.linear_fit``."""
    tc = t - t.mean()
    stt = float(tc @ tc)
    if stt <= 0:
        return float(y.mean()), 0.0, y - y.mean()
    b = float(tc @ (y - y.mean())) / stt
    a = float(y.mean() - b * t.mean())
    return a, b, y - (a + b * t)


def _deviation0(c, t_blocks, y_blocks):
    """The 0.5 ``DeviationCriterion.evaluate`` of ``c``."""
    mean = float(y_blocks.mean())
    if c.model == "constant":
        resid = y_blocks - mean
    else:
        resid = _linear_fit0(t_blocks, y_blocks)[2]
    dev = float(np.max(np.abs(resid)))
    sigma = float(c.noise(mean)) if callable(c.noise) else float(c.noise)
    if c.relative:
        scale = max(abs(mean), c.scale_floor)
        if not scale > 0:
            return _Result0(c.name, float("inf"), 0.0, c.limit, c.k)
        dev = dev / scale
    return _Result0(c.name, dev, sigma, c.limit, c.k)


class _Reference:
    """A built-in criterion as 0.5 judged it: its results take ``_Result0``'s properties, and a deviation criterion
    is evaluated by ``_deviation0``. Every window a deviation criterion sees is kept in ``windows``."""

    def __init__(self, criterion, windows):
        self._c, self._windows = criterion, windows

    def __getattr__(self, name):
        return getattr(self._c, name)

    def evaluate(self, t_blocks, y_blocks, **kw):
        if isinstance(self._c, rs.DeviationCriterion):
            self._windows.append((self._c, t_blocks.copy(), y_blocks.copy()))
            return _deviation0(self._c, t_blocks, y_blocks)
        r = self._c.evaluate(t_blocks, y_blocks, **kw)
        return _Result0(r.name, r.statistic, r.uncertainty, r.limit, r.k, r.details)


class _StepChange0(rs.Disturbance):
    """The 0.5 ``StepChange``."""

    def __init__(self, signal, threshold, m=10, name=None):
        self.signal, self.threshold, self.m = signal, threshold, m
        self.name = name or f"{signal}_step"
        self.reset()

    def reset(self):
        from collections import deque  # pylint: disable=import-outside-toplevel
        self._hist = deque(maxlen=2 * self.m)

    def update(self, t, sample, values):
        self._hist.append(values[self.signal])
        if len(self._hist) < 2 * self.m:
            return False
        h = np.fromiter(self._hist, dtype=float, count=2 * self.m)
        return bool(abs(h[self.m :].mean() - h[: self.m].mean()) > self.threshold)


def _run0(detector, df, t):
    """The 0.5 ``batch.run(detector, df, t, with_results=True)``, with the 0.5 ``Verdict.eta`` and ``limiting``."""
    detail_cols = _detail_columns(detector)
    detector.reset()
    cols = _unique_columns(df)
    data = df.to_numpy()
    rows = []
    for i in range(len(df)):
        v = detector.update(float(t[i]), dict(zip(cols, data[i])))
        eta = max((r.ratio for r in v.results.values()), default=np.nan)
        limiting = max(v.results.values(), key=lambda r: r.ratio).name if v.results else None
        row = {"label": v.label.value, "eta": eta, "limiting": limiting,
               "since_disturbance_s": v.since_disturbance_s, "forced_by": v.forced_by,
               "reason": None if v.reason is None else v.reason.value, "valid": v.valid,
               "window_start_s": v.window_start_s}
        for name, r in v.results.items():
            row[f"ratio_{name}"] = r.ratio
            for key, value in r.details.items():
                row[f"detail_{name}_{key}"] = value
        rows.append(row)
    columns = list(RUN_COLUMNS) + [f"ratio_{c.name}" for c in detector.criteria] + detail_cols
    out = pd.DataFrame(rows, index=df.index, columns=columns)
    floats = ["eta", "since_disturbance_s", "window_start_s"] + [c for c in columns
                                                                 if c.startswith(("ratio_", "detail_"))]
    out[floats] = out[floats].astype(float)
    out["valid"] = out["valid"].astype(bool)
    out["t_s"] = np.asarray(t, dtype=float)
    return out


# --- a record that reaches every reason -------------------------------------

UNREADABLE = 9e300
"""A reading of ``q`` that its signal cannot read."""


def _q(sample):
    q = sample["q"]
    return "unreadable" if q == UNREADABLE else q


def _p_noise(level):
    """Relative noise of ``p``'s block mean, unknown (NaN) near full power, so some windows are invalid."""
    return float("nan") if level > 1000.0 else 0.001


def _record():
    """Seeded samples ``(df, t)``: power ``p`` that rises, decays below its floor and comes back, a rod ``r``
    that steps and ramps, a temperature ``q`` approaching its asymptote, with missing rows, an unreadable one, a skipped
    time step and a gap long enough to clear the history."""
    rng = np.random.default_rng(180)
    n = 6000
    target = np.zeros(n)
    target[800:2500], target[3400:] = 1000.0, 500.0
    p = np.zeros(n)
    for i in range(1, n):  # first-order response with a 30 s time constant
        p[i] = p[i - 1] + (target[i] - p[i - 1]) / 30.0
    p[3400:] += np.arange(n - 3400) * 0.01  # a slow drift
    p = p * (1 + rng.normal(0, 0.003, n))
    r = np.full(n, 10.0)
    r[1000:], r[2500:], r[3200:] = 15.0, 5.0, 12.0
    r[4000:5000] += np.arange(1000) * 0.004
    r[5000:] += 4.0
    r[5200:] += 1.0  # a move too small for the step detector: the line fit fails across it
    r = r + rng.normal(0, 0.03, n)
    q = 20.0 + 30.0 * (1 - np.exp(-np.clip(np.arange(n) - 300.0, 0, None) / 100.0))
    noise = np.zeros(n)
    for i in range(1, n):  # AR(1) sensor noise
        noise[i] = 0.9 * noise[i - 1] + rng.normal(0, 0.02)
    q = q + noise
    df = pd.DataFrame({"p": p, "r": r, "q": q})
    for rows in ([1500], [1600, 1601, 1602], range(1800, 1804), range(1700, 1706), [4100]):
        df.iloc[list(rows)] = np.nan
    df.loc[2200, "q"] = UNREADABLE
    t = np.arange(n, dtype=float)
    t[2000:] += 2.0  # two samples skipped
    t[5500:] += 50.0  # a gap that clears the history
    return df, t


def _detector(reference, windows=None):
    criteria = [rs.DeviationCriterion("p", 5.0, 0.5, name="p_abs"),
                rs.DeviationCriterion("p", 0.01, _p_noise, relative=True, name="p_rel"),
                rs.DeviationCriterion("r", 0.3, 0.02, model="linear", name="r_lin"),
                rs.DeviationCriterion("p", 0.02, 0.001, model="linear", relative=True, name="p_rel_lin"),
                rs.DeviationCriterion("r", 0.02, 0.002, relative=True, scale_floor=1.0, name="r_rel"),
                rs.DriftCriterion("q", 0.002, name="q_drift"),
                rs.BoundCriterion("q", 45.0, 55.0, name="q_bound"),
                rs.AsymptoteCriterion("q", 1.0, 600.0, n_grid=30, name="q_asym")]
    step = _StepChange0 if reference else rs.StepChange
    disturbances = [step("r", 2.0, 5), step("p", 100.0, 8), rs.EventLog([(4500.0, 10.0)])]
    if reference:
        criteria = [_Reference(c, windows) for c in criteria]
    timing = rs.Timing(block_s=10.0, window_s=200.0, settle_s=20.0, hold_s=30.0, hold_when_undecided=True,
                       max_gap_s=5.0, min_block_coverage=0.7)
    return rs.Detector([rs.Signal("p", "p"), rs.Signal("r", "r"), rs.Signal("q", _q)], criteria, disturbances,
                       [rs.FloorLatch("p", 1.0)], timing)


@pytest.fixture(scope="module")
def runs():
    """``(new, reference, windows)``: the runs of this version and of the 0.5 code on ``_record``, and the windows
    the reference deviation criteria saw."""
    df, t = _record()
    windows: list = []
    return rs.batch.run(_detector(False), df, t, with_results=True), _run0(_detector(True, windows), df, t), windows


def _bits(x):
    return np.asarray(x, dtype=float).view(np.int64)


def test_the_record_reaches_every_reason_and_crosses_every_limit(runs):
    _, old, _ = runs
    assert set(old["reason"]) == {r.value for r in rs.Reason}
    assert set(old["label"]) == {label.value for label in Label}
    for c in _detector(False).criteria:
        ratio = old[f"ratio_{c.name}"]
        assert (ratio < 1).any() and ((ratio > 1) & np.isfinite(ratio)).any(), c.name
    assert np.isinf(old.loc[300:799, "ratio_p_rel"]).any()  # an invalid result from a zero scale (p is 0)
    assert np.isinf(old.loc[1000:2499, "ratio_p_rel"]).any()  # and from a NaN noise (p near 1000)
    assert (old.loc[:1999, "reason"] == "disturbance").any()  # a step detector's: the event log's event is later
    # Ties for the worst ratio: the limiting criterion is the first of them, neither the first nor the last criterion.
    ratios = old[[f"ratio_{c.name}" for c in _detector(False).criteria]]
    tied = (ratios.eq(old["eta"], axis=0).sum(axis=1) >= 2) & (old["limiting"] == "p_rel")
    assert tied.any() and (ratios.loc[tied, "ratio_p_rel_lin"] == old.loc[tied, "eta"]).all()


def test_run_gives_the_0_5_output_bit_for_bit(runs):
    new, old, _ = runs
    pd.testing.assert_frame_equal(new, old, check_exact=True)
    for col in new.columns:
        if new[col].dtype == float:
            assert np.array_equal(_bits(new[col]), _bits(old[col])), col


def test_line_fit_and_deviation_criteria_give_the_0_5_values_bit_for_bit(runs):
    _, _, windows = runs
    complete = _detector(False)._t_blocks  # pylint: disable=protected-access
    assert any(not np.array_equal(t, complete) for _, t, _ in windows)  # partial blocks move their times
    assert {c.name for c, _, _ in windows} == {"p_abs", "p_rel", "r_lin", "p_rel_lin", "r_rel"}
    for c, t, y in windows:
        fit, fit0 = rs.stats.linear_fit(t, y), _linear_fit0(t, y)
        assert [_bits(v).tolist() for v in fit] == [_bits(v).tolist() for v in fit0]
        same_time = np.full_like(t, 2.0)  # no spread in time: the fit's flat branch
        assert [_bits(v).tolist() for v in rs.stats.linear_fit(same_time, y)] == \
            [_bits(v).tolist() for v in _linear_fit0(same_time, y)]
        r, r0 = c.evaluate(t, y), _deviation0(c, t, y)
        assert (r.name, r.details) == (r0.name, r0.details)
        assert _bits([r.statistic, r.uncertainty, r.limit, r.k]).tolist() == \
            _bits([r0.statistic, r0.uncertainty, r0.limit, r0.k]).tolist()


def _fit_and_warnings(fit, t, y):
    """``fit(t, y)`` as bits, and the distinct messages of the warnings it gave (0.5 took some means twice, so it
    gave some warnings twice)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fit(t, y)
    return [_bits(v).tolist() for v in out], sorted({str(w.message) for w in caught})


@pytest.mark.parametrize("t, y", [
    # int64 times and values whose sums overflow int64; their means take a float64 sum
    (np.arange(4, dtype=np.int64) * 2**61, np.array([2**62, 2**62, 2**62, 1], dtype=np.int64)),
    (np.arange(5, dtype=np.float32), np.array([1, 2, 2, 3, 5], dtype=np.float32)),
    (np.empty(0), np.empty(0)),  # no samples: mean's own warning
    # float64 off any grid, whose sums round differently in another order
    (np.random.default_rng(3).normal(0, 1e3, 20), np.random.default_rng(4).normal(0, 1, 20)),
], ids=["int64", "float32", "empty", "float64"])
def test_line_fit_gives_the_0_5_values_and_warnings_on_other_arrays(t, y):
    assert _fit_and_warnings(rs.stats.linear_fit, t, y) == _fit_and_warnings(_linear_fit0, t, y)


def test_a_deviation_criterion_gives_the_0_5_values_on_a_masked_window():
    """A masked array's mean counts its unmasked items only."""
    c = rs.DeviationCriterion("x", 1.0, 0.1)
    t, y = np.arange(4.0), np.ma.array([1.0, 50.0, 2.0, 4.0], mask=[0, 1, 0, 0])
    r, r0 = c.evaluate(t, y), _deviation0(c, t, y)
    assert _bits([r.statistic, r.uncertainty]).tolist() == _bits([r0.statistic, r0.uncertainty]).tolist()
    assert r0.statistic == pytest.approx(5 / 3)  # |4 - 7/3|: the unmasked mean


def test_a_step_detector_raises_on_a_value_it_cannot_store_and_leaves_it_out():
    """A decision on #180: 0.5.1 kept such a value, and raised only while its full history held it."""
    early = rs.StepChange("x", 1.0, m=2)
    assert early.update(0.0, {}, {"x": 0.0}) is False
    with pytest.raises(ValueError):  # at once, though the history is not yet full
        early.update(1.0, {}, {"x": "x"})
    # Left out: the next samples are judged as if it never came, not with what its slot held before (5.0).
    step, clean = rs.StepChange("x", 1.0, m=1), rs.StepChange("x", 1.0, m=1)
    assert [step.update(0.0, {}, {"x": 5.0}), step.update(1.0, {}, {"x": 0.0})] == [False, True]
    with pytest.raises(ValueError):
        step.update(2.0, {}, {"x": "x"})
    flags = [step.update(float(i), {}, {"x": v}) for i, v in enumerate([0.0, 0.0, 5.0], start=3)]
    assert flags == [clean.update(float(i), {}, {"x": v}) for i, v in enumerate([5.0, 0.0, 0.0, 0.0, 5.0])][2:]
    assert flags == [False, False, True]


# Pairs of consecutive values that compare equal: inf then np.float64(inf) (their reprs in invalid_reason differ
# under numpy 2), -0.0 then 0.0 (a limit's repr), and 3.0 then np.float64(3.0) (another type).
VALUES = [math.inf, np.float64(math.inf), math.nan, 1.5, -0.0, 0.0, -1.0, 3.0, np.float64(3.0), 0.25]


@pytest.mark.parametrize("field", ["statistic", "uncertainty", "limit", "k"])
def test_derived_properties_follow_a_reassigned_field(field):
    r = rs.CriterionResult("c", 0.5, 0.1, 1.0, 2.0)
    for value in VALUES:
        _ = r.invalid_reason, r.ratio, r.outcome  # read before the field changes
        setattr(r, field, value)
        fresh = _Result0("c", r.statistic, r.uncertainty, r.limit, r.k)
        assert r.invalid_reason == fresh.invalid_reason, value
        assert r.outcome == fresh.outcome, value
        assert _bits([r.ratio, r.upper, r.lower]).tolist() == _bits([fresh.ratio, fresh.upper, fresh.lower]).tolist()


class _Offset(rs.CriterionResult):
    """A subclass whose ``upper`` also reads an attribute that is not a field."""

    offset = 0.0

    @property
    def upper(self):
        return self.statistic + self.k * self.uncertainty + self.offset


def test_a_subclass_works_out_its_derived_properties_on_every_read():
    r = _Offset("c", 0.5, 0.1, 1.0, 2.0)
    assert (r.outcome, r.ratio < 1) == (Outcome.PASS, True)
    r.offset = 0.6  # upper 1.3, lower 0.3: straddles the limit
    assert (r.outcome, r.ratio > 1) == (Outcome.UNDECIDED, True)

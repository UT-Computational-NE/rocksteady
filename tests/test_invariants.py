"""
Cross-cutting behavioural invariants (#28).

The focused regressions for #14-#21 stay with their fixes; these check
properties that must hold for any record:

- renaming or reordering uniquely named criteria cannot change acceptance;
- permuting channels cannot change a derived signal's missingness;
- a change of units scales slopes and levels and leaves time constants alone;
- an omitted timestamp and an explicit missing row behave the same;
- backfill stays inside the certifying window, never crossing a
  disturbance or a reset;
- empty and non-empty outputs share a schema.
"""

import itertools

import numpy as np
import pandas as pd
import pytest

import rocksteady as rs
from rocksteady import stats
from rocksteady.fit import fit_exp, fit_first_order, fit_poly
from rocksteady.metrics import stats_by_segment, stats_by_state
from rocksteady.states import Segment, State

SEEDS = range(6)


def _record(seed, n=1200):
    """Two channels: plateaus, a first-order step and AR(1) noise, with a few dropouts."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    step_at = int(rng.integers(250, 450))
    base = np.where(t < step_at, 10.0, 10.0 + 3.0 * (1 - np.exp(-(t - step_at) / 25.0)))
    e = np.zeros((n, 2))
    for i in range(1, n):
        e[i] = 0.6 * e[i - 1] + rng.normal(0, 0.01, 2)
    x = base[:, None] + e + [0.0, 0.2]
    for j in rng.integers(0, n, size=4):
        x[j, rng.integers(0, 2)] = np.nan
    return pd.DataFrame({"a": x[:, 0], "b": x[:, 1]}), [(float(step_at), 0.0)]


def _criteria(scale=1.0):
    return [
        rs.DriftCriterion("mean", 1e-3 * scale, name="drift"),
        rs.DeviationCriterion("mean", 0.05 * scale, 0.01 * scale, name="deviation"),
        rs.DeviationCriterion("spread", 0.1 * scale, 0.01 * scale, model="constant", name="spread"),
    ]


def _detector(criteria, events, **timing):
    kw = dict(block_s=10, window_s=200, max_gap_s=3, settle_s=20, hold_s=10)
    kw.update(timing)
    signals = [rs.factory.signal("mean", builder="mean", cols=["a", "b"]),
               rs.factory.signal("spread", builder="spread", cols=["a", "b"])]
    return rs.Detector(signals, criteria, [rs.EventLog(events)], timing=rs.Timing(**kw))


@pytest.mark.parametrize("seed", SEEDS)
def test_renaming_or_reordering_criteria_cannot_change_acceptance(seed):
    df, events = _record(seed)
    want = rs.batch.run(_detector(_criteria(), events), df)["label"].tolist()
    assert "steady" in want
    for perm in itertools.permutations(range(3)):
        crit = [_criteria()[i] for i in perm]
        for c, new in zip(crit, ("z", "y", "x")):
            c.name = new
        assert rs.batch.run(_detector(crit, events), df)["label"].tolist() == want


@pytest.mark.parametrize("builder", ["mean", "spread"])
def test_channel_permutation_cannot_change_missingness(builder):
    rng = np.random.default_rng(0)
    cols = ["c0", "c1", "c2", "c3"]
    rows = [dict(zip(cols, r)) for r in rng.normal(size=(200, 4))]
    for r in rows[::7]:
        r[cols[int(rng.integers(0, 4))]] = float("nan")
    for r in rows[::11]:
        del r[cols[int(rng.integers(0, 4))]]  # an unreadable channel
    values = {}
    for perm in itertools.permutations(cols):
        sig = rs.factory.signal("s", builder=builder, cols=list(perm))
        out = []
        for r in rows:
            try:
                out.append(sig.value(r))
            except KeyError:
                out.append(float("nan"))
        values[perm] = np.array(out, dtype=float)
    ref = values[tuple(cols)]
    for v in values.values():
        np.testing.assert_array_equal(np.isnan(v), np.isnan(ref))
        np.testing.assert_allclose(v, ref, rtol=1e-12)


@pytest.mark.parametrize("scale, offset", [(1000.0, 0.0), (1.8, 32.0), (-2.0, 5.0)])
def test_units_scale_slopes_and_levels_not_time_constants(scale, offset):
    rng = np.random.default_rng(1)
    t = 1e6 + np.arange(400.0)  # a large time origin: nothing may depend on it
    s = t - t[0]
    noise = rng.normal(0, 0.01, 400)
    for fit, y in [(lambda tt, yy: fit_poly(tt, yy, 1), 2.0 + 0.003 * s + noise),
                   (fit_first_order, 5.0 - 3.0 * np.exp(-s / 60.0) + noise),
                   (fit_exp, 1.0 + 2.0 * np.exp(-s / 80.0) + noise)]:
        a, b = fit(t, y), fit(t, scale * y + offset)
        for k, v in a.params.items():
            if k in ("tau", "T", "lambda"):
                assert b.params[k] == pytest.approx(v, rel=1e-6)
                assert b.u[k] == pytest.approx(a.u[k], rel=1e-5)
            elif k in ("level", "y0", "y_inf", "c"):
                assert b.params[k] == pytest.approx(scale * v + offset, rel=1e-6, abs=1e-9 * abs(offset))
            else:  # slope, curvature, amplitude
                assert b.params[k] == pytest.approx(scale * v, rel=1e-6)
                assert b.u[k] == pytest.approx(abs(scale) * a.u[k], rel=1e-5)
    y = 0.002 * s + noise
    b0, u0, k0 = stats.slope_ar1(t, y)
    b1, u1, k1 = stats.slope_ar1(t, scale * y + offset)
    assert b1 == pytest.approx(scale * b0, rel=1e-9) and u1 == pytest.approx(abs(scale) * u0, rel=1e-9) and k1 == k0


@pytest.mark.parametrize("seed", SEEDS)
def test_detector_labels_are_unit_invariant(seed):
    df, events = _record(seed)
    want = rs.batch.run(_detector(_criteria(), events), df)["label"].tolist()
    for scale, offset in [(1000.0, 0.0), (1.8, 32.0)]:
        got = rs.batch.run(_detector(_criteria(scale), events), df * scale + offset)["label"].tolist()
        assert got == want


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("max_gap_s", [0, 3])
def test_omitted_timestamps_and_missing_rows_are_equivalent(seed, max_gap_s):
    df, events = _record(seed)
    rng = np.random.default_rng(100 + seed)
    t = np.arange(len(df), dtype=float)
    gone = np.zeros(len(df), dtype=bool)
    for start in rng.integers(50, len(df) - 10, size=5):
        gone[start:start + int(rng.integers(1, 4))] = True
    explicit = df.copy()
    explicit.loc[gone] = np.nan
    det = _detector(_criteria(), events, max_gap_s=max_gap_s)
    out_e = rs.batch.run(det, explicit, t=t)
    out_o = rs.batch.run(det, df[~gone], t=t[~gone])
    # Every observed sample gets the same label either way.
    assert out_e["label"][~gone].tolist() == out_o["label"].tolist()
    assert not out_e["valid"][gone].any()
    # The same steady intervals, in time (an explicit record may start or end on a missing row).
    steady_e = [s for s in rs.segments_from(det, out_e) if s.state == State.STEADY]
    steady_o = [s for s in rs.segments_from(det, out_o) if s.state == State.STEADY]
    t_o = t[~gone]
    assert len(steady_e) == len(steady_o)
    for se, so in zip(steady_e, steady_o):
        observed = t[se.start:se.stop][~gone[se.start:se.stop]]
        assert (observed[0], observed[-1]) == (t_o[so.start], t_o[so.stop - 1])


@pytest.mark.parametrize("seed", SEEDS)
def test_backfill_stays_inside_the_certifying_window(seed):
    df, events = _record(seed)
    df.iloc[600:606] = np.nan  # a gap longer than max_gap_s: a reset
    det = _detector(_criteria(), events)
    out = rs.batch.run(det, df)
    labels, reasons = out["label"].to_numpy(), out["reason"].to_numpy()
    assert "gap_reset" in set(reasons) and "disturbed" in set(labels)
    n_checked = 0
    for s in rs.segments_from(det, out):
        if s.state != State.STEADY:
            continue
        # The first passing row (a bridged missing sample may join two accepted runs into one segment).
        a = s.start + int(np.flatnonzero(np.isin(labels[s.slice], ["steady", "holding"]))[0])
        if out["forced_by"].iloc[a] is None:
            assert out["t_s"].iloc[s.start] >= out["window_start_s"].iloc[a]
        span = slice(s.start, a)
        assert "disturbed" not in set(labels[span]) and "gap_reset" not in set(reasons[span])
        n_checked += 1
    assert n_checked >= 2  # before the reset and after it


def test_empty_and_nonempty_outputs_share_a_schema():
    df, events = _record(0)
    det = _detector(_criteria(), events)
    for with_results in (False, True):
        full = rs.batch.run(det, df, with_results=with_results)
        empty = rs.batch.run(det, df.iloc[:0], with_results=with_results)
        assert list(full.columns) == list(empty.columns)
        numeric = ["eta", "since_disturbance_s", "valid", "window_start_s", "t_s"]
        numeric += [c for c in full.columns if c.startswith("ratio_")]
        assert full[numeric].dtypes.tolist() == empty[numeric].dtypes.tolist()
    a, b = np.arange(10.0), np.arange(10.0) + 1
    segs = [Segment(0, 10, State.STEADY)]
    assert list(stats_by_segment(a, b, segs, dt=1.0).columns) == list(stats_by_segment(a, b, [], dt=1.0).columns)
    assert list(stats_by_state(a, b, segs, dt=1.0).columns) == list(stats_by_state(a, b, [], dt=1.0).columns)
    comp = rs.report.compare_report(a, b, segs, dt=1.0).by_interval
    comp_empty = rs.report.compare_report(a, b, [], dt=1.0).by_interval
    assert list(comp.columns) == list(comp_empty.columns)
    rep = rs.report.fit_report({"x": a}, segs, {"x": 1}, dt=1.0).by_interval
    rep_empty = rs.report.fit_report({"x": a}, [], {"x": 1}, dt=1.0).by_interval
    common = [c for c in rep_empty.columns if c != "signal"]
    assert list(rep.columns[:len(common)]) == common  # fitted parameters follow, by model

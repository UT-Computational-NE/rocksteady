"""Tests of intervals by state and of batch.signal_values."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

import rocksteady as rs
from rocksteady import Label, State

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_regression import actuator_case  # noqa: E402

ST, HO, SE, DI, IN = "steady", "holding", "settling", "disturbed", "indeterminate"


def _states(segs, n):
    """Per-sample state codes from segments, checking coverage and order on the way."""
    out = np.empty(n, dtype=object)
    pos = 0
    for s in segs:
        assert s.start == pos and s.stop > s.start
        out[s.slice] = s.state.value
        pos = s.stop
    assert pos == n
    return out


def test_three_states_and_label_mapping():
    labels = [IN, IN, ST, ST, DI, SE, HO, HO, SE, IN, HO, ST, ST]
    segs = rs.segments(labels)
    assert [(s.start, s.stop, s.state) for s in segs] == [
        (0, 2, State.INDETERMINATE),
        (2, 4, State.STEADY),
        (4, 9, State.TRANSIENT),  # holding that never confirmed is transient
        (9, 10, State.INDETERMINATE),
        (10, 13, State.STEADY),  # holding that confirmed is steady
    ]
    assert len(segs[2]) == 5 and segs[2].slice == slice(4, 9)


def test_accepts_label_enums_and_series():
    labels = [Label.INDETERMINATE, Label.STEADY, Label.SETTLING]
    want = [State.INDETERMINATE, State.STEADY, State.TRANSIENT]
    assert [s.state for s in rs.segments(labels)] == want
    assert [s.state for s in rs.segments(pd.Series([str(x) for x in labels]))] == want


def test_random_labels_are_covered_without_overlap():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(1, 60))
        labels = rng.choice([ST, HO, SE, DI, IN], size=n, p=[0.4, 0.15, 0.2, 0.1, 0.15])
        for kw in ({}, {"backfill_samples": 5}, {"min_samples": 3}, {"backfill_samples": 4, "min_samples": 4}):
            segs = rs.segments(labels, **kw)
            _states(segs, n)
            # Maximal runs, except that adjacent steady segments are distinct accepted intervals.
            assert all(a.state != b.state or a.state == State.STEADY for a, b in zip(segs, segs[1:]))
            steady = [(s.start, s.stop) for s in segs if s.state == State.STEADY]
            assert [b for _, b in steady] == [b for _, b in rs.batch.accepted_intervals(labels)]


def test_backfill_extends_steady_and_keeps_accepted_intervals_separate():
    labels = [IN] * 5 + [ST] * 3 + [SE] * 2 + [ST] * 3
    segs = rs.segments(labels, backfill_samples=3)
    # First run backfills from 5 to 2; the second would reach 7 but stops at the first's end (8).
    assert [(s.start, s.stop, s.state.value) for s in segs] == [(0, 2, IN), (2, 8, ST), (8, 13, ST)]
    segs = rs.segments([SE] * 10 + [ST] * 2, backfill_samples=100)
    assert [(s.start, s.stop, s.state) for s in segs] == [(0, 12, State.STEADY)]  # bounded by the record start


def test_backfill_shorter_than_gap_leaves_transient():
    labels = [ST] * 3 + [SE] * 4 + [ST] * 2
    segs = rs.segments(labels, backfill_samples=2)
    assert [(s.start, s.stop, s.state.value) for s in segs] == [(0, 3, ST), (3, 5, "transient"), (5, 9, ST)]


def test_min_samples_merges_short_non_steady_runs():
    labels = [SE] * 6 + [IN] + [SE] * 3 + [ST] * 4 + [IN] * 2 + [ST] * 3
    segs = rs.segments(labels, min_samples=2)
    assert [(s.start, s.stop, s.state.value) for s in segs] == [
        (0, 10, "transient"),  # the lone indeterminate sample joins the transient run around it
        (10, 14, ST),
        (14, 16, IN),  # long enough, kept
        (16, 19, ST),
    ]
    # A short run with no non-steady neighbour is kept.
    segs = rs.segments([ST] * 3 + [IN] + [ST] * 3, min_samples=5)
    assert [s.state.value for s in segs] == [ST, IN, ST]


@pytest.mark.parametrize("label, state", [(ST, State.STEADY), (SE, State.TRANSIENT), (IN, State.INDETERMINATE)])
def test_single_state_records(label, state):
    assert rs.segments([label] * 7) == [rs.Segment(0, 7, state)]
    assert rs.segments([]) == []


def test_argument_checks():
    with pytest.raises(ValueError):
        rs.segments([ST], backfill_samples=-1)
    with pytest.raises(ValueError):
        rs.segments([ST], min_samples=0)


def test_golden_actuator_case():
    det, df = actuator_case()
    out = rs.batch.run(det, df)
    labels = out["label"]
    plain = rs.segments(labels)
    steady = [(s.start, s.stop) for s in plain if s.state == State.STEADY]
    assert steady == rs.batch.accepted_intervals(labels)
    filled = rs.segments_from(det, out)
    forced = out["forced_by"].notna().to_numpy()
    disturbed = np.flatnonzero(labels.to_numpy() == DI)
    window = det.timing.window_samples - 1  # samples before the first steady one in its certifying window
    prev = 0
    n_forced = 0
    for (a, b), s in zip(steady, [s for s in filled if s.state == State.STEADY]):
        if forced[a]:
            assert (s.start, s.stop) == (a, b)  # an override-forced run is not backfilled
            n_forced += 1
        else:
            last_dist = disturbed[disturbed < a].max() + 1 if (disturbed < a).any() else 0
            assert (s.start, s.stop) == (max(a - window, prev, last_dist), b)
        prev = b
    assert n_forced >= 1  # the floor latch forces at least one steady run in this case
    assert {s.state for s in plain} == set(State)
    unfilled = rs.segments_from(det, out, backfill=False)
    assert [(s.start, s.stop, s.state) for s in unfilled] == [(s.start, s.stop, s.state) for s in plain]
    # Provenance survives segmentation: a segment is forced exactly when an override forced one of its labels.
    assert [s.forced for s in unfilled] == [bool(forced[s.slice].any()) for s in unfilled]
    assert any(s.forced for s in unfilled) and not any(s.forced for s in plain)
    with pytest.warns(UserWarning, match="labels only"):
        rs.segments_from(det, labels)


def test_backfill_skips_forced_runs_and_stops_at_disturbances():
    labels = [SE] * 10 + [ST] * 3 + [DI] * 2 + [SE] * 3 + [ST] * 4
    forced = [False] * 10 + [True] * 3 + [False] * 9
    segs = rs.segments(labels, backfill_samples=6, forced=forced)
    assert [(s.start, s.stop, s.state.value) for s in segs] == [
        (0, 10, "transient"),
        (10, 13, ST),  # forced: not backfilled
        (13, 15, "transient"),  # the disturbance stays transient
        (15, 22, ST),  # backfilled to just after the disturbance, not the full 6 samples
    ]
    with pytest.raises(ValueError, match="one entry per label"):
        rs.segments(labels, forced=[True])


def test_backfill_covers_exactly_the_certifying_window():
    det = rs.Detector([rs.Signal("x", "x")], [rs.DeviationCriterion("x", limit=1, noise=0)],
                      timing=rs.Timing(block_s=1, window_s=3))
    out = rs.batch.run(det, pd.DataFrame({"x": [10, 0, 0, 0]}))
    assert out["label"].tolist() == [IN, IN, SE, ST]
    # The window that certified sample 3 is samples 1-3; the rejected sample 0 stays out.
    assert [(s.start, s.stop, s.state) for s in rs.segments_from(det, out)] == [
        (0, 1, State.INDETERMINATE), (1, 4, State.STEADY)]
    # After a reset (a missing sample), the missing sample is not backfilled either.
    out = rs.batch.run(det, pd.DataFrame({"x": [0, 0, 0, np.nan, 0, 0, 0]}))
    segs = rs.segments_from(det, out)
    assert [(s.start, s.stop, s.state) for s in segs][-1] == (4, 7, State.STEADY)


def test_signal_values():
    det, df = actuator_case()
    values = rs.batch.signal_values(det, df)
    assert list(values.columns) == ["level", "act"] and values.index.equals(df.index)
    i = 10
    sample = df.iloc[i].to_dict()
    assert values["level"].iloc[i] == det.signals[0].value(sample)
    assert values["act"].iloc[i] == sample["act"]
    assert np.isnan(values["level"].iloc[900])  # the dropout row
    floor = values["level"].iloc[2300]
    assert floor == 0.0  # below the floor, the signal maps to zero


# -- #17: backfill follows time when samples are missing from the record ------------


def _gap_detector(**timing):
    kw = dict(block_s=2, window_s=6, max_gap_s=2, min_block_coverage=0.5)
    kw.update(timing)
    return rs.Detector([rs.Signal("x", "x")], [rs.DeviationCriterion("x", 1, 0)], timing=rs.Timing(**kw))


def _steady_times(segs, t):
    return [(float(t[s.start]), float(t[s.stop - 1])) for s in segs if s.state == State.STEADY]


def test_backfill_stays_inside_the_certifying_window_with_omitted_timestamps():
    det = _gap_detector()
    t = np.array([0., 1., 2., 3., 4., 5., 8., 9., 10.])
    df = pd.DataFrame({"x": [50., 50., 50., 50., 50., 0., 0., 0., 0.]})
    out = rs.batch.run(det, df, t=t)
    assert out["label"].iloc[-1] == "steady"
    steady = [s for s in rs.segments_from(det, out) if s.state == State.STEADY]
    # The window passing at t = 10 covers t = 5..10 (with t = 6, 7 missing): the first stored row is index 5.
    assert steady[0].start == 5 and (df["x"].iloc[steady[0].start:] == 0).all()


@pytest.mark.parametrize("missing", [[6, 7], [6], [4, 7], [3, 6], [5, 7], [3, 6, 9], [9], [9, 12], [12, 13], [15]])
def test_omitted_timestamps_and_explicit_nan_rows_agree(missing):
    """Same steady intervals, start and end; gaps before and inside the run, inside blocks and at the end.

    A tolerated missing row inside a steady run is bridged (#24), as an omitted timestamp is."""
    full_t = np.arange(16.0)
    x = np.where(full_t < 2, 50.0, 0.0)
    explicit = pd.DataFrame({"x": np.where(np.isin(full_t, missing), np.nan, x)})
    keep = ~np.isin(full_t, missing)
    omitted = pd.DataFrame({"x": x[keep]})
    det = _gap_detector(window_s=6, max_gap_s=3, min_block_coverage=0.5)
    out_e = rs.batch.run(det, explicit, t=full_t)
    out_o = rs.batch.run(det, omitted, t=full_t[keep])
    assert set(out_e["reason"][np.isin(full_t, missing)]) == {"missing"}
    te = _steady_times(rs.segments_from(det, out_e), full_t)
    to = _steady_times(rs.segments_from(det, out_o), full_t[keep])
    assert to and len(to) == len(te)  # low block coverage may split both the same way
    # Same physical window start; the explicit record may start on a NaN row the omitted one does not have.
    assert to[0][0] == min(tt for tt in full_t[keep] if tt >= te[0][0])
    assert [b for _, b in to] == [b for _, b in te]
    assert [a for a, _ in to[1:]] == [a for a, _ in te[1:]]


def test_time_backfill_arguments():
    labels = [SE, SE, ST]
    with pytest.raises(ValueError, match="both t and backfill_s"):
        rs.segments(labels, t=[0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="one entry per label"):
        rs.segments(labels, t=[0.0, 1.0], backfill_s=1.0)
    assert rs.segments(labels, t=[0.0, 5.0, 6.0], backfill_s=1.5)[-1].start == 1  # time, not count
    assert rs.segments(labels, backfill_samples=1)[-1].start == 1
    # Labels without times still backfill by count.
    det = _gap_detector()
    out = rs.batch.run(det, pd.DataFrame({"x": np.zeros(10)}))
    assert rs.segments_from(det, out["label"]) == rs.segments_from(det, out.drop(columns="t_s"))


@pytest.mark.parametrize("origin", [0.0, 1.7e9, -3.2e8, 1e12])
def test_time_backfill_does_not_depend_on_the_clock_origin(origin):
    labels = ["settling"] * 5 + ["steady"]
    segs = rs.segments(labels, t=origin + np.arange(6.0), backfill_s=2.0)
    assert segs[-1].start == 3  # rows 3, 4, 5: ages 2, 1, 0 s
    # Through the detector: the rejected value 10 (row 2) is outside the certifying window.
    det = rs.Detector([rs.Signal("x", "x")], [rs.DeviationCriterion("x", 1, 0)],
                      timing=rs.Timing(block_s=1, window_s=3))
    df = pd.DataFrame({"x": [50.0, 30.0, 10.0, 0.0, 0.0, 0.0]})
    out = rs.batch.run(det, df, t=origin + np.arange(6.0))
    steady = [s for s in rs.segments_from(det, out) if s.state == State.STEADY]
    assert [(s.start, s.stop) for s in steady] == [(3, 6)]
    with pytest.raises(ValueError, match="backfill_tol_s"):
        rs.segments(labels, t=origin + np.arange(6.0), backfill_s=2.0, backfill_tol_s=-1.0)


@pytest.mark.parametrize("origin", [0.0, 1.7e9, 1e12])
def test_time_backfill_boundary_and_rounding(origin):
    labels = ["settling"] * 5 + ["steady"]
    t = origin + np.arange(6.0)
    # Zero tolerance keeps the sample exactly backfill_s old (t = 3 for a window ending at t = 5).
    assert rs.segments(labels, t=t, backfill_s=2.0, backfill_tol_s=0.0)[-1].start == 3
    # Also with times that are not exact in binary (0.1 s spacing; ulps of 2e-7 s at epoch times).
    t01 = origin + 0.1 * np.arange(6.0)
    assert rs.segments(labels, t=t01, backfill_s=0.2, backfill_tol_s=0.0)[-1].start == 3
    # Rounding to whole periods (default tolerance: half the spacing): a sample 0.4 period beyond
    # the window rounds into it, one 0.6 period beyond does not.
    near = origin + np.array([0.0, 1.0, 2.0, 2.6, 4.0, 5.0])  # t[3] is 2.4 s old: 0.4 beyond 2.0
    far = origin + np.array([0.0, 1.0, 2.0, 2.4, 4.0, 5.0])   # t[3] is 2.6 s old: 0.6 beyond 2.0
    assert rs.segments(labels, t=near, backfill_s=2.0, backfill_tol_s=0.5)[-1].start == 3
    assert rs.segments(labels, t=far, backfill_s=2.0, backfill_tol_s=0.5)[-1].start == 4

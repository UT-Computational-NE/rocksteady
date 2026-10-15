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
    window = det.timing.window_samples
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
    assert rs.segments_from(det, out, backfill=False) == plain
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

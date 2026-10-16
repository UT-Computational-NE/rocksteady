"""Run-output columns, the sampling contract, and time metadata in statistics (#22)."""

import numpy as np
import pandas as pd
import pytest

import rocksteady as rs
from rocksteady.metrics import error_stats, split_at_time_gaps, stats_by_state
from rocksteady.states import Segment, State

NAN = float("nan")


def _det(**timing):
    kw = {"block_s": 2, "window_s": 6, "max_gap_s": 2, **timing}
    return rs.Detector([rs.Signal("x", "x")], [rs.DeviationCriterion("x", 1, 0)], timing=rs.Timing(**kw))


def test_valid_and_certifying_window_columns():
    det = _det()
    t = 100.0 + np.arange(14.0)
    x = np.zeros(14)
    x[9] = NAN
    out = rs.batch.run(det, pd.DataFrame({"x": x}), t=t)
    assert list(out.columns[:8]) == list(rs.batch.RUN_COLUMNS)
    assert out["valid"].dtype == bool and out["valid"].tolist() == [i != 9 for i in range(14)]
    w = det.timing.evaluated_samples
    examined = out["window_start_s"].notna()
    assert not examined[:5].any() and not examined[9]  # no window before one window of history, none on a NaN row
    np.testing.assert_allclose(out["window_start_s"][examined], out["t_s"][examined] - (w - 1))
    # The certifying window of the first steady row is where segments_from backfills to.
    first = int(np.argmax(out["label"].to_numpy() == "steady"))
    steady = [s for s in rs.segments_from(det, out) if s.state == State.STEADY][0]
    assert out["t_s"].iloc[steady.start] == out["window_start_s"].iloc[first]


def test_rows_map_one_to_one_with_the_input_index():
    det = _det()
    idx = pd.date_range("2026-01-01", periods=12, freq="1s")
    df = pd.DataFrame({"x": np.zeros(12)}, index=idx).drop(idx[[4, 5]])  # omitted timestamps
    out = rs.batch.run(det, df)
    assert out.index.equals(df.index)
    assert out["t_s"].tolist() == [0, 1, 2, 3, 6, 7, 8, 9, 10, 11]


@pytest.mark.parametrize("times, match", [
    ([0.0, 1.0, 1.4], "sampled faster than sample_period_s"),
    ([0.0, 0.4, 1.0], "sampled faster than sample_period_s"),
])
def test_faster_sampling_is_rejected(times, match):
    det = _det()
    with pytest.raises(ValueError, match=match):
        for t in times:
            det.update(t, {"x": 0.0})


def test_grid_tolerance():
    jittery = [0.0, 1.3, 2.1, 3.0, 4.6]  # steps 1.3, 0.8, 0.9, 1.6 periods
    det = _det()
    for t in jittery:  # the default (0.5) rounds every step to the grid
        det.update(t, {"x": 0.0})
    det = _det(grid_tolerance=0.2)
    with pytest.raises(ValueError, match="off the sampling grid"):
        for t in jittery:
            det.update(t, {"x": 0.0})
    det = _det(grid_tolerance=0.2, max_gap_s=5)
    for t in [0.0, 1.1, 2.0, 5.1, 6.0]:  # within 0.2 of the grid, including a gap of two samples
        det.update(t, {"x": 0.0})
    for bad in (0.0, 0.6, NAN):
        with pytest.raises(ValueError, match="grid_tolerance"):
            rs.Timing(block_s=1, window_s=3, grid_tolerance=bad)


def test_split_at_time_gaps():
    t = np.array([0, 1, 2, 5, 6, 7, 8, 20, 21.0])
    assert split_at_time_gaps([(0, 9)], t, 1.0) == [(0, 3), (3, 7), (7, 9)]
    assert split_at_time_gaps([(1, 5), (6, 9)], t, 1.0) == [(1, 3), (3, 5), (6, 7), (7, 9)]
    assert split_at_time_gaps([(0, 9)], None, 1.0) == [(0, 9)]
    assert split_at_time_gaps([(0, 9)], t, 20.0) == [(0, 9)]  # steps under 1.5 dt are not gaps


def test_autocorrelation_respects_omitted_timestamps():
    rng = np.random.default_rng(3)
    n = 400
    a = np.cumsum(rng.normal(size=n)) * 0.05 + rng.normal(size=n)
    b = np.zeros(n)
    t = np.arange(n, dtype=float)
    t[200:] += 50  # 50 samples omitted between rows 199 and 200
    with_t = error_stats(a, b, dt=1.0, t=t)
    as_pieces = error_stats(a, b, dt=1.0, pieces=[(0, 200), (200, n)])
    np.testing.assert_equal(with_t.as_dict(), as_pieces.as_dict())
    assert with_t.tau_int_s != error_stats(a, b, dt=1.0).tau_int_s
    segs = [Segment(0, 300, State.STEADY), Segment(300, n, State.TRANSIENT)]
    by_t = stats_by_state(a, b, segs, dt=1.0, t=t)
    three = error_stats(a, b, dt=1.0, pieces=[(0, 200), (200, 300), (300, n)])  # time gap and segment boundary
    assert by_t.loc["all", "tau_int_s"] == three.tau_int_s
    with pytest.raises(ValueError, match="one entry per sample"):
        error_stats(a, b, dt=1.0, t=t[:-1])


def test_reports_use_the_record_times():
    rng = np.random.default_rng(4)
    n = 300
    t = np.arange(n, dtype=float)
    t[150:] += 30
    a = rng.normal(size=n).cumsum() * 0.1
    segs = [Segment(0, n, State.STEADY)]
    rep = rs.report.compare_report(a, np.zeros(n), segs, t=t, dt=1.0)
    want = error_stats(a, np.zeros(n), dt=1.0, pieces=[(0, 150), (150, n)])
    assert rep.by_state.loc[("a", "all"), "tau_int_s"] == want.tau_int_s

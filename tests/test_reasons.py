"""Verdict reasons, the missing-sample hold policy and its combinations (#24)."""

import numpy as np
import pandas as pd
import pytest

import rocksteady as rs
from rocksteady import Label, Reason
from rocksteady.states import State

NAN = float("nan")


def _det(overrides=(), disturbances=(), noise=0.0, **timing):
    kw = {"block_s": 2, "window_s": 6, "max_gap_s": 2, **timing}
    return rs.Detector([rs.Signal("x", "x")], [rs.DeviationCriterion("x", 1, noise)], disturbances, overrides,
                       rs.Timing(**kw))


def _run(det, x, t=None):
    out = rs.batch.run(det, pd.DataFrame({"x": np.asarray(x, dtype=float)}), t=t)
    return list(out["label"]), list(out["reason"])


def test_every_verdict_has_a_reason():
    det = _det(settle_s=2, disturbances=[rs.EventLog([(9.0, 0.0)])])
    x = [0.0] * 8 + [5.0] + [0.0] * 12
    labels, reasons = _run(det, x)
    assert reasons[:5] == ["insufficient_history"] * 5
    assert reasons[5:7] == ["settling_time"] * 2 and labels[5:7] == ["indeterminate"] * 2  # start of record
    assert (labels[7], reasons[7]) == ("steady", "certified")
    assert (labels[8], reasons[8]) == ("settling", "failed")  # the step is in the window
    assert (labels[9], reasons[9]) == ("disturbed", "disturbance")
    assert set(reasons[10:17]) == {"settling_time"} and set(labels[10:17]) == {"settling"}
    assert (labels[17], reasons[17]) == ("steady", "certified")
    assert None not in reasons


def test_missing_unreadable_and_gap_reset():
    det = _det(block_s=1, window_s=3, max_gap_s=1, min_block_coverage=1.0)
    for t in range(4):
        det.update(float(t), {"x": 0.0})
    v = det.update(4.0, {"x": NAN})
    assert (v.label, v.reason) == (Label.INDETERMINATE, Reason.MISSING)
    v = det.update(5.0, {"x": 0.0})
    assert v.reason == Reason.LOW_COVERAGE and not v.results  # a block of one sample is empty
    det.reset()
    for t in range(4):
        det.update(float(t), {"x": 0.0})
    assert det.update(4.0, {}).reason == Reason.UNREADABLE
    assert det.update(5.0, {"x": "?"}).reason == Reason.GAP_RESET  # two missing samples exceed max_gap_s = 1
    assert det.update(6.0, {"x": 0.0}).reason == Reason.INSUFFICIENT_HISTORY
    # A jump in time beyond max_gap_s: the first sample after it reports the reset.
    assert det.update(20.0, {"x": 0.0}).reason == Reason.GAP_RESET
    assert det.update(21.0, {"x": 0.0}).reason == Reason.INSUFFICIENT_HISTORY


def test_statistical_overlap_and_invalid_results():
    rng = np.random.default_rng(0)
    # Noise straddling the limit: the uncertainty overlaps it, so some windows are undecided.
    labels, reasons = _run(_det(block_s=1, window_s=3, noise=0.6), rng.normal(0, 0.6, 200))
    assert "undecided" in reasons
    assert {lab for lab, r in zip(labels, reasons) if r == "undecided"} == {"indeterminate"}
    labels, reasons = _run(_det(block_s=1, window_s=3, noise=lambda level: -1.0), np.zeros(6))
    assert reasons[3:] == ["invalid_result"] * 3 and set(labels[3:]) == {"indeterminate"}


# -- the hold clock pauses over missing samples, whatever the undecided policy ------

POLICIES = [(False, False), (False, True), (True, False), (True, True)]


@pytest.mark.parametrize("hold_when_undecided, counts", POLICIES)
@pytest.mark.parametrize("form", ["nan_rows", "omitted"])
def test_missing_samples_pause_the_hold_clock(hold_when_undecided, counts, form):
    det = _det(hold_s=2, hold_when_undecided=hold_when_undecided, undecided_counts_toward_hold=counts)
    clean, _ = _run(det, np.zeros(14))
    assert clean.index("steady") == 7  # passing windows at 5, 6, 7: STEADY after hold_s = 2 more
    t = np.arange(14.0)
    if form == "nan_rows":
        labels, reasons = _run(det, np.where(t == 6, NAN, 0.0))
        assert (labels[6], reasons[6]) == ("indeterminate", "missing")
        assert reasons[:6] + reasons[7:] == ["insufficient_history"] * 5 + ["certified"] * 8
    else:
        keep = t != 6
        labels, _ = _run(det, np.zeros(13), t=t[keep])
        labels = labels[:6] + ["(omitted)"] + labels[6:]
    # The clock neither resets nor advances over the missing sample: STEADY comes one sample later.
    assert labels[5] == "holding" and labels[7] == "holding" and labels.index("steady") == 8


@pytest.mark.parametrize("hold_when_undecided, counts, first_steady", [
    (False, False, None), (False, True, None), (True, False, 13), (True, True, 11)])
def test_missing_samples_then_low_coverage(hold_when_undecided, counts, first_steady):
    """Two missing samples (paused), then windows with an empty block (the undecided policy applies)."""
    x = np.zeros(16)
    x[[6, 7]] = NAN
    det = _det(hold_s=4, hold_when_undecided=hold_when_undecided, undecided_counts_toward_hold=counts)
    labels, reasons = _run(det, x)
    assert reasons[6:12] == ["missing", "missing", "certified", "low_coverage", "certified", "low_coverage"]
    held = labels[9]
    assert held == ("holding" if hold_when_undecided else "indeterminate")
    assert (labels.index("steady") if "steady" in labels else None) == first_steady


# -- overrides and disturbances across gaps ----------------------------------------


def _latch_record():
    """Certified steady at 5, a step to 3 (settling), then below the floor of 1 at sample 12 (latched)."""
    return np.r_[np.full(8, 5.0), np.full(4, 3.0), np.full(10, 0.5)]


def test_forced_labels_across_a_tolerated_gap():
    det = _det(overrides=[rs.FloorLatch("x", 1.0)])
    x = _latch_record()
    x[15] = NAN
    out = rs.batch.run(det, pd.DataFrame({"x": x}))
    assert list(out["reason"][12:15]) == ["forced"] * 3 and set(out["forced_by"][12:15]) == {"floor_latch"}
    assert (out["label"][15], out["reason"][15]) == ("indeterminate", "missing") and pd.isna(out["forced_by"][15])
    # The latch survives a tolerated gap, so the label is still forced after it.
    assert out["reason"][16] == "forced" and out["label"][16] == "steady"
    steady = [s for s in rs.segments_from(det, out) if s.state == State.STEADY]
    # Certified, then forced; the missing sample is bridged, and the forced run is not backfilled.
    assert [s.forced for s in steady[:-1]] == [False] * (len(steady) - 1) and steady[-1].forced
    assert (steady[-1].start, steady[-1].stop) == (12, 22)


def test_a_gap_reset_releases_overrides():
    det = _det(overrides=[rs.FloorLatch("x", 1.0)], max_gap_s=0)
    x = _latch_record()
    x[15] = NAN
    labels, reasons = _run(det, x)
    assert reasons[12:15] == ["forced"] * 3 and reasons[15] == "gap_reset"
    assert "forced" not in reasons[16:]  # the latch state was cleared with the history
    assert reasons[16:21] == ["insufficient_history"] * 5


@pytest.mark.parametrize("form", ["nan_rows", "omitted"])
def test_an_event_during_a_gap_is_not_lost(form):
    det = _det(disturbances=[rs.EventLog([(6.5, 0.0)])])
    t = np.arange(14.0)
    gap = (t == 6) | (t == 7)
    if form == "nan_rows":
        labels, reasons = _run(det, np.where(gap, NAN, 0.0), t=t)
        after = 8
    else:
        labels, reasons = _run(det, np.zeros(12), t=t[~gap])
        after = 6
    assert (labels[after], reasons[after]) == ("disturbed", "disturbance")


def test_reason_column_and_provenance_in_reports():
    det = _det(overrides=[rs.FloorLatch("x", 1.0)])
    df = pd.DataFrame({"x": _latch_record()})
    out = rs.batch.run(det, df)
    assert set(out["reason"]) <= {r.value for r in Reason}
    rep = rs.batch.fit_report(det, df, {"x": 0}, labels=out)
    table = rep.by_interval
    assert "forced" in table.columns and table["forced"].any() and not table["forced"].all()
    assert "steady (forced)" in rep.to_string()
    comp = rs.report.compare_report(df["x"].to_numpy(), df["x"].to_numpy() + 1, rs.segments_from(det, out), dt=1.0)
    assert comp.by_interval["forced"].any()

"""Synthetic tests of the detector state machine and the criteria."""

import numpy as np
import pytest

import rocksteady as rs
from rocksteady import Label

RNG = np.random.default_rng(1)


def feed(det, samples):
    return [det.update(float(i), s).label for i, s in enumerate(samples)]


def flat(n, level=5.0, noise=0.05):
    return [{"x": level + RNG.normal(0, noise)} for _ in range(n)]


def make(criteria, disturbances=(), overrides=(), **timing):
    kw = dict(window_s=120.0, block_s=10.0)
    kw.update(timing)
    return rs.Detector([rs.Signal("x", "x"), rs.Signal("z", lambda s: s.get("z", 0.0))],
                       criteria, disturbances, overrides, rs.Timing(**kw))


def test_flat_signal_becomes_steady_after_window():
    det = make([rs.DeviationCriterion("x", limit=0.5, noise=0.02)])
    labels = feed(det, flat(300))
    assert labels[:119] == [Label.INDETERMINATE] * 119
    assert all(l == Label.STEADY for l in labels[119:])


def test_step_change_disturbs_then_settles_for_window():
    det = make([rs.DeviationCriterion("x", limit=0.5, noise=0.02)],
               disturbances=[rs.StepChange("z", threshold=1.0, m=5)])
    samples = [{"x": 5.0, "z": 0.0 if i < 200 else 10.0} for i in range(500)]
    labels = feed(det, samples)
    first_dist = labels.index(Label.DISTURBED)
    assert 200 <= first_dist <= 201
    last_dist = max(i for i, l in enumerate(labels) if l == Label.DISTURBED)
    assert all(l == Label.SETTLING for l in labels[last_dist + 1 : last_dist + 120])
    assert labels[last_dist + 120] == Label.STEADY


def test_linear_model_admits_drift_constant_model_does_not():
    ramp = [{"x": 0.05 * i} for i in range(300)]
    lin = make([rs.DeviationCriterion("x", limit=0.5, noise=0.01, model="linear")])
    con = make([rs.DeviationCriterion("x", limit=0.5, noise=0.01, model="constant")])
    assert feed(lin, ramp)[-1] == Label.STEADY
    assert feed(con, ramp)[-1] == Label.SETTLING


def test_three_way_decision():
    r = rs.CriterionResult("c", statistic=1.0, uncertainty=0.1, limit=1.5, k=3)
    assert r.outcome == rs.Outcome.PASS
    r = rs.CriterionResult("c", statistic=2.0, uncertainty=0.1, limit=1.5, k=3)
    assert r.outcome == rs.Outcome.FAIL
    r = rs.CriterionResult("c", statistic=1.4, uncertainty=0.1, limit=1.5, k=3)
    assert r.outcome == rs.Outcome.UNDECIDED
    assert r.ratio == pytest.approx(1.7 / 1.5)


def test_undecided_is_indeterminate_unless_held():
    crit = [rs.DeviationCriterion("x", limit=0.1, noise=0.05, k=3)]  # noise straddles the limit
    labels = feed(make(crit), flat(300, noise=0.0))
    assert labels[-1] == Label.INDETERMINATE
    held = make(crit, hold_when_undecided=True)
    labels = feed(held, flat(300, noise=0.0))
    assert labels[-1] == Label.INDETERMINATE  # nothing was ever decided


def test_hold_time_gives_holding_then_steady():
    det = make([rs.DeviationCriterion("x", limit=0.5, noise=0.02)], hold_s=60.0)
    labels = feed(det, flat(400, noise=0.0))
    assert labels[119] == Label.HOLDING
    assert labels[178] == Label.HOLDING
    assert labels[179] == Label.STEADY


def test_event_log_resets_sequence():
    det = make([rs.DeviationCriterion("x", limit=0.5, noise=0.02)],
               disturbances=[rs.EventLog([(250.0, 0.0)])], settle_s=30.0)
    labels = feed(det, flat(500, noise=0.0))
    assert labels[249] == Label.STEADY
    assert labels[250] == Label.DISTURBED
    assert all(l == Label.SETTLING for l in labels[251:400])  # window + settle = 150 s
    assert labels[400] == Label.STEADY


def test_override_forces_label_and_disturbance_wins():
    class Always(rs.Override):
        name = "always"

        def apply(self, ctx):
            return Label.STEADY

    det = make([rs.DeviationCriterion("x", limit=0.5, noise=0.02)], overrides=[Always()],
               disturbances=[rs.StepChange("z", 1.0, m=5)])
    samples = [{"x": 5.0, "z": 0.0 if i < 50 else 10.0} for i in range(100)]
    labels = feed(det, samples)
    assert labels[0] == Label.STEADY and labels[49] == Label.STEADY
    assert Label.DISTURBED in labels[50:60]
    assert det.update(100.0, {"x": 5.0, "z": 10.0}).forced_by == "always"


def test_nan_resets_history():
    det = make([rs.DeviationCriterion("x", limit=0.5, noise=0.02)])
    labels = feed(det, flat(200, noise=0.0))
    assert labels[-1] == Label.STEADY
    assert det.update(200.0, {"x": float("nan")}).label == Label.INDETERMINATE
    assert det.update(201.0, {"x": 5.0}).label == Label.INDETERMINATE


def test_drift_criterion_slope_equivalence():
    t = np.arange(40) * 30.0
    flat_y = 20.0 + RNG.normal(0, 0.01, 40)
    r = rs.DriftCriterion("x", limit_per_s=0.01 / 60).evaluate(t, flat_y)
    assert r.outcome == rs.Outcome.PASS
    r = rs.DriftCriterion("x", limit_per_s=0.01 / 60).evaluate(t, 20.0 + 0.05 / 60 * t)
    assert r.outcome == rs.Outcome.FAIL
    assert r.statistic == pytest.approx(0.05 / 60)


def test_batch_matches_streaming_and_intervals():
    import pandas as pd

    det = make([rs.DeviationCriterion("x", limit=0.5, noise=0.02)], hold_s=30.0)
    df = pd.DataFrame(flat(300, noise=0.0))
    out = rs.batch.run(det, df)
    assert list(out["label"].iloc[119:149]) == ["holding"] * 30
    assert rs.batch.accepted_intervals(out["label"]) == [(119, 300)]
    assert rs.batch.intervals(out["label"], Label.STEADY) == [(149, 300)]


# -- 0.2.0 behaviour -----------------------------------------------------------


def _feed_t(det, ts, samples):
    return [det.update(float(t), s).label for t, s in zip(ts, samples)]


def test_ratio_is_infinite_for_non_finite_results_and_eta_is_order_free():
    bad = rs.CriterionResult("bad", statistic=float("nan"), uncertainty=0.1, limit=1.0, k=2)
    good = rs.CriterionResult("good", statistic=0.5, uncertainty=0.1, limit=1.0, k=2)
    assert bad.ratio == np.inf and bad.outcome == rs.Outcome.UNDECIDED
    for results in ({"bad": bad, "good": good}, {"good": good, "bad": bad}):
        v = rs.Verdict(0.0, Label.INDETERMINATE, results)
        assert v.eta == np.inf and v.limiting == "bad"


def test_window_must_suit_criteria():
    with pytest.raises(ValueError, match="at least 4 blocks"):
        make([rs.DriftCriterion("x", limit_per_s=1.0)], window_s=30.0)
    make([rs.DeviationCriterion("x", limit=0.5, noise=0.02)], window_s=30.0)  # 3 blocks is enough here


def test_startup_settle_is_indeterminate_not_settling():
    det = make([rs.DeviationCriterion("x", limit=0.5, noise=0.02)], settle_s=30.0)
    labels = feed(det, flat(200, noise=0.0))
    assert set(labels[:149]) == {Label.INDETERMINATE}
    assert labels[149] == Label.STEADY


class _PassUnlessHigh(rs.Criterion):
    """Passes on a low last block, undecided on a high one."""

    name, signal = "toggle", "x"

    def evaluate(self, t_blocks, y_blocks):
        stat = float("nan") if y_blocks[-1] > 0.5 else 0.0
        return rs.CriterionResult(self.name, stat, 0.0, 1.0, 1.0)


@pytest.mark.parametrize("counts", [True, False])
def test_undecided_counts_toward_hold(counts):
    det = make([_PassUnlessHigh()], window_s=30.0, hold_s=50.0, hold_when_undecided=True,
               undecided_counts_toward_hold=counts)
    labels = feed(det, [{"x": 0.0 if i < 50 else 1.0} for i in range(120)])
    assert labels[29] == Label.HOLDING
    assert labels[100] == (Label.STEADY if counts else Label.HOLDING)


def test_single_nan_resets_by_default_and_is_tolerated_with_max_gap():
    samples = flat(300, noise=0.0)
    samples[200] = {"x": float("nan")}
    strict = feed(make([rs.DeviationCriterion("x", limit=0.5, noise=0.02)]), samples)
    assert strict[199] == Label.STEADY and strict[201] == Label.INDETERMINATE
    lenient = feed(make([rs.DeviationCriterion("x", limit=0.5, noise=0.02)], max_gap_s=5.0), samples)
    assert lenient[200] == Label.INDETERMINATE  # the missing sample itself
    assert lenient[201] == Label.STEADY


def test_gap_longer_than_max_gap_resets():
    samples = flat(300, noise=0.0)
    for i in range(200, 206):
        samples[i] = {"x": float("nan")}
    labels = feed(make([rs.DeviationCriterion("x", limit=0.5, noise=0.02)], max_gap_s=5.0), samples)
    assert labels[199] == Label.STEADY
    assert set(labels[206:300]) == {Label.INDETERMINATE}  # history restarted at 206; window needs 120


def test_block_coverage_makes_window_undecided():
    samples = flat(300, noise=0.0)
    samples[150] = {"x": float("nan")}
    det = make([rs.DeviationCriterion("x", limit=0.5, noise=0.02)], max_gap_s=5.0, min_block_coverage=0.95)
    labels = feed(det, samples)
    assert labels[151] == Label.INDETERMINATE and labels[269] == Label.INDETERMINATE
    assert labels[270] == Label.STEADY  # the gap has left the window


def test_timestamp_gaps():
    crit = [rs.DeviationCriterion("x", limit=0.5, noise=0.02)]
    ts = list(range(200)) + [203.0] + list(range(204, 300))  # three samples missing
    samples = flat(len(ts), noise=0.0)
    assert _feed_t(make(crit), ts, samples)[200] == Label.INDETERMINATE  # default: reset
    assert _feed_t(make(crit, max_gap_s=5.0), ts, samples)[200] == Label.STEADY
    ts_long = list(range(200)) + [260.0] + list(range(261, 357))
    assert _feed_t(make(crit, max_gap_s=5.0), ts_long, samples)[200] == Label.INDETERMINATE
    with pytest.raises(ValueError, match="increase"):
        _feed_t(make(crit), [0.0, 1.0, 1.0], flat(3))


def test_batch_uses_datetime_index():
    import pandas as pd

    idx = pd.date_range("2026-01-01", periods=300, freq="1s")
    idx = idx.append(pd.date_range("2026-01-01 01:00", periods=300, freq="1s"))  # one-hour gap
    df = pd.DataFrame(flat(600, noise=0.0), index=idx)
    out = rs.batch.run(make([rs.DeviationCriterion("x", limit=0.5, noise=0.02)]), df)
    assert out["t_s"].iloc[300] == 3600.0
    assert out["label"].iloc[299] == "steady" and out["label"].iloc[300] == "indeterminate"
    assert out["label"].iloc[419] == "steady"


def test_bound_criterion_uncertainty_accounts_for_autocorrelation():
    n = 60
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = 0.9 * e[i - 1] + RNG.normal()
    r = rs.BoundCriterion("x", low=-10.0, high=10.0).evaluate(np.arange(n) * 30.0, e)
    assert r.uncertainty > e.std(ddof=1) / np.sqrt(n) * 1.5

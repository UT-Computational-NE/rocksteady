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
    assert r.test == rs.Test.PASS
    r = rs.CriterionResult("c", statistic=2.0, uncertainty=0.1, limit=1.5, k=3)
    assert r.test == rs.Test.FAIL
    r = rs.CriterionResult("c", statistic=1.4, uncertainty=0.1, limit=1.5, k=3)
    assert r.test == rs.Test.UNDECIDED
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
    assert r.test == rs.Test.PASS
    r = rs.DriftCriterion("x", limit_per_s=0.01 / 60).evaluate(t, 20.0 + 0.05 / 60 * t)
    assert r.test == rs.Test.FAIL
    assert r.statistic == pytest.approx(0.05 / 60)


def test_batch_matches_streaming_and_intervals():
    import pandas as pd

    det = make([rs.DeviationCriterion("x", limit=0.5, noise=0.02)], hold_s=30.0)
    df = pd.DataFrame(flat(300, noise=0.0))
    out = rs.batch.run(det, df)
    assert list(out["label"].iloc[119:149]) == ["holding"] * 30
    assert rs.batch.accepted_intervals(out["label"]) == [(119, 300)]
    assert rs.batch.intervals(out["label"], Label.STEADY) == [(149, 300)]

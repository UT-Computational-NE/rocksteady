"""Configuration, provider-result and numerical-input validation (#23)."""

import numpy as np
import pytest

import rocksteady as rs
from rocksteady import Label, Outcome
from rocksteady.fit import check_model_options, fit_exp, fit_first_order, fit_poly, piecewise_fit, validate_models
from rocksteady.metrics import stats_by_segment, stats_by_state
from rocksteady.states import Segment, State, check_segments

NAN, INF = float("nan"), float("inf")


def _det(criteria=None, disturbances=(), overrides=(), **timing):
    kw = {"block_s": 1, "window_s": 3, **timing}
    return rs.Detector([rs.Signal("x", "x")], criteria or [rs.DeviationCriterion("x", 1, 0)], disturbances,
                       overrides, rs.Timing(**kw))


@pytest.mark.parametrize("opt, value, match", [
    ("sample_period_s", NAN, "sample_period_s must be finite"), ("block_s", INF, "block_s must be finite"),
    ("window_s", -1.0, "window_s must be > 0"), ("settle_s", NAN, "settle_s must be finite"),
    ("hold_s", -5.0, "hold_s must be >= 0"), ("max_gap_s", INF, "max_gap_s must be finite"),
    ("min_block_coverage", 0.0, "min_block_coverage"), ("sample_period_s", True, "must be a number"),
])
def test_timing_rejects_invalid_values(opt, value, match):
    with pytest.raises(ValueError, match=match):
        rs.Timing(**{"block_s": 1.0, "window_s": 3.0, opt: value})


@pytest.mark.parametrize("make, match", [
    (lambda: rs.Signal("", "x"), "Signal name must be a non-empty string"),
    (lambda: rs.Signal("x", ""), "source must be a column name or a callable"),
    (lambda: rs.DeviationCriterion("x", -1, 0), "limit must be > 0"),
    (lambda: rs.DeviationCriterion("x", 1, -0.1), "noise must be >= 0"),
    (lambda: rs.DeviationCriterion("x", 1, 0, k=NAN), "k must be finite"),
    (lambda: rs.DeviationCriterion("x", 1, 0, model="cubic"), "model must be"),
    (lambda: rs.DriftCriterion("x", 0.0), "limit_per_s must be > 0"),
    (lambda: rs.DriftCriterion("x", 1.0, k=-2.0), "k must be >= 0"),
    (lambda: rs.BoundCriterion("x", 5, 5), "low .* must be < high"),
    (lambda: rs.BoundCriterion("x", -INF, 5), "low must be finite"),
    (lambda: rs.StepChange("x", 1.0, m=0), "m must be a positive integer"),
    (lambda: rs.StepChange("x", 1.0, m=2.5), "m must be a positive integer"),
    (lambda: rs.StepChange("x", -1.0), "threshold must be > 0"),
    (lambda: rs.EventLog([(1.0,)]), "event 0 must be"),
    (lambda: rs.EventLog([(1.0, -2.0)]), "event 0 duration must be >= 0"),
    (lambda: rs.EventLog([(NAN, 0.0)]), "event 0 start must be finite"),
    (lambda: rs.FloorLatch("x", 0.0), "floor must be > 0"),
    (lambda: rs.FloorLatch("x", 1.0, release_factor=0.5), "release_factor must be >= 1"),
])
def test_components_reject_invalid_arguments(make, match):
    with pytest.raises(ValueError, match=match):
        make()


def test_detector_checks_references_and_names():
    with pytest.raises(ValueError, match="disturbance 'z_step' refers to unknown signal 'z'"):
        _det(disturbances=[rs.StepChange("z", 1.0)])
    with pytest.raises(ValueError, match="override 'floor_latch' refers to unknown signal 'y'"):
        _det(overrides=[rs.FloorLatch("y", 1.0)])

    class Nameless(rs.Criterion):  # a custom criterion that forgot its name
        name, signal = "", "x"

        def evaluate(self, t_blocks, y_blocks):
            return rs.CriterionResult("", 0.0, 0.0, 1.0, 1.0)

    with pytest.raises(ValueError, match="Nameless name must be a non-empty string"):
        _det([Nameless()])
    _det(disturbances=[rs.EventLog([(1, 0)])])  # no signal reference: fine


def test_specs_are_validated_the_same_way():
    base = {"timing": {"block_s": 1, "window_s": 3}, "signals": ["x"]}
    with pytest.raises(ValueError, match="limit must be > 0"):
        rs.build_detector({**base, "criteria": [{"type": "deviation", "signal": "x", "limit": -1, "noise": 0}]})
    with pytest.raises(ValueError, match="window_s must be finite"):
        rs.build_detector({**base, "timing": {"block_s": 1, "window_s": float("inf")}})
    with pytest.raises(ValueError, match="refers to unknown signal 'y'"):
        rs.build_detector({**base, "criteria": [{"type": "deviation", "signal": "x", "limit": 1, "noise": 0}],
                           "disturbances": [{"type": "step", "signal": "y", "threshold": 1}]})


def test_invalid_criterion_results_never_certify():
    assert rs.CriterionResult("bad", 2, -1, 1, 3).outcome == Outcome.UNDECIDED  # used to PASS
    assert "uncertainty is negative" in rs.CriterionResult("bad", 2, -1, 1, 3).invalid_reason
    assert rs.CriterionResult("bad", 2, 1, 1, -3).outcome == Outcome.UNDECIDED  # negative k
    assert rs.CriterionResult("bad", 0.1, 0.1, 1, NAN).ratio == np.inf
    assert rs.CriterionResult("ok", 0.5, 0.1, 1, 2).invalid_reason is None
    # A callable noise model returning a negative value cannot certify a window.
    det = _det([rs.DeviationCriterion("x", 1, lambda level: -0.5)])
    labels = [det.update(t, {"x": 1.0}).label for t in range(5)]
    assert Label.STEADY not in labels and labels[-1] == Label.INDETERMINATE


def test_fit_bounds_and_options_are_checked_before_data():
    t = np.arange(10.0)
    for kwargs, match in [({"T_min": -1}, "T_min must be > 0"), ({"T_min": 0.0}, "T_min must be > 0"),
                          ({"T_min": NAN}, "T_min must be finite"), ({"T_min": 50.0, "T_max": 5.0}, "T_min .* <= T_max")]:
        with pytest.raises(ValueError, match=match):
            fit_exp(t, t, **kwargs)
    with pytest.raises(ValueError, match="tau_min .* <= tau_max"):
        fit_first_order(t, t, tau_min=10.0, tau_max=1.0)
    with pytest.raises(ValueError, match="first_order: uncertainty"):
        fit_first_order(t, t, uncertainty="hac")
    with pytest.raises(ValueError, match="poly: order"):
        fit_poly(t, t, True)
    with pytest.raises(ValueError, match="invalid options"):
        check_model_options("exp", {"tau_min": 1.0})
    with pytest.raises(ValueError, match=r"model for 'power' in state 'transient': exp: T_min must be > 0"):
        validate_models({"power": {"steady": 0, "transient": {"model": "exp", "T_min": -1}}})
    with pytest.raises(ValueError, match=r"model for 'power' in state 'steady': poly: invalid options"):
        validate_models({"power": {"steady": {"model": "poly", "degree": 2}, "transient": 1}})
    validate_models({"power": {"steady": 0, "transient": {"model": "first_order", "tau_max": 1e4}}})


def test_segments_are_checked():
    segs = [Segment(0, 4, State.STEADY), Segment(2, 6, State.TRANSIENT)]
    with pytest.raises(ValueError, match="overlaps"):
        check_segments(segs, 6)
    with pytest.raises(ValueError, match="overlaps"):
        stats_by_state(np.zeros(6), np.zeros(6), segs, dt=1.0)  # used to count 8 samples out of 6
    with pytest.raises(ValueError, match="outside a record of 6"):
        check_segments([Segment(4, 9, State.STEADY)], 6)
    partial = [Segment(1, 3, State.STEADY)]
    assert check_segments(partial, 6) == partial  # partial coverage is allowed
    with pytest.raises(ValueError, match="full coverage"):
        check_segments(partial, 6, require_full=True)
    assert stats_by_segment(np.zeros(6), np.ones(6), partial, dt=1.0)["n"].tolist() == [2]
    with pytest.raises(ValueError, match="overlaps"):
        piecewise_fit(np.arange(6.0), np.zeros(6), segs, 0)

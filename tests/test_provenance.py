"""Provenance, stable evaluation and serialization, and the AR(1) sample-spacing assumption (#26)."""

import json

import numpy as np
import pandas as pd
import pytest

import rocksteady as rs
from rocksteady import _ar1
from rocksteady.fit import (evaluate, fit_exp, fit_first_order, fit_poly, normalize_model, residual_phi,
                            sandwich_cov)
from rocksteady.states import Segment, State

T0 = 1.7e9  # an epoch-like clock: references must survive a large time origin


def _fits():
    t = T0 + np.arange(200.0)
    s = t - T0
    rng = np.random.default_rng(0)
    return {
        "poly": (fit_poly(t, 3.0 + 0.01 * s - 1e-4 * s**2 + rng.normal(0, 0.01, 200), 2), t),
        "first_order": (fit_first_order(t, 5.0 - 4.0 * np.exp(-s / 40.0) + rng.normal(0, 0.01, 200)), t),
        "exp_decay": (fit_exp(t, 1.0 + 3.0 * np.exp(-s / 30.0) + rng.normal(0, 0.01, 200)), t),
        # steep growth: the amplitude at the start underflows, the stable form does not
        "exp_growth": (fit_exp(t, 1.0 + np.exp((s - 199.0) / 0.25), T_min=0.1), t),
    }


@pytest.mark.parametrize("name", ["poly", "first_order", "exp_decay", "exp_growth"])
def test_predict_reproduces_the_fit_and_survives_json(name):
    r, t = _fits()[name]
    np.testing.assert_array_equal(r.predict(t), r.fitted)
    record = json.loads(json.dumps(r.to_dict()))
    np.testing.assert_array_equal(evaluate(record["model"], record["evaluation"], t), r.fitted)
    assert record["param_order"] == list(r.params) and len(record["cov"]) == len(r.params)
    assert record["uncertainty"] == "ar1" and record["t_ref"] == r.t_ref
    # Evaluation between and beyond samples, on the same clock.
    assert np.isfinite(r.predict([t[0] + 0.5, t[-1] + 0.1])).all()


def test_reference_times_and_stable_growth_amplitude():
    fits = _fits()
    assert fits["poly"][0].t_ref == T0 + 99.5  # interval centre
    assert fits["first_order"][0].t_ref == T0 and fits["exp_decay"][0].t_ref == T0  # first valid sample
    growth = fits["exp_growth"][0]
    assert growth.params["A"] == 0.0  # exp(-199 / 0.25) underflows: display only
    assert growth.evaluation["t_ref"] == T0 + 199.0 and growth.evaluation["B"] == pytest.approx(1.0, rel=1e-6)
    with pytest.raises(ValueError, match="no evaluation form"):
        fit_poly(np.arange(3.0), np.full(3, np.nan)).predict([0.0])


def test_normalize_model_fills_defaults():
    assert normalize_model(0) == {"model": "poly", "order": 0, "uncertainty": "ar1"}
    assert normalize_model({"steady": 1, "transient": {"model": "exp", "T_max": 50}}, "indeterminate") == {
        "model": "exp", "T_min": None, "T_max": 50, "uncertainty": "ar1"}


# -- the AR(1) sample-spacing assumption ------------------------------------------


def test_ar1_apply_with_steps_matches_dense_positions():
    rng = np.random.default_rng(1)
    pos = np.cumsum(rng.integers(1, 4, size=60)).astype(float)
    x = rng.normal(size=(60, 2))
    for phi in (0.3, 0.9):
        dense = phi ** np.abs(np.subtract.outer(pos, pos))
        np.testing.assert_allclose(_ar1.ar1_apply(x, phi, np.diff(pos)), dense @ x, rtol=1e-12)
    np.testing.assert_array_equal(_ar1.ar1_apply(x, 0.5, np.ones(59)), _ar1.ar1_apply(x, 0.5))
    assert _ar1.gap_steps(np.arange(5.0), 1.0) is None
    np.testing.assert_array_equal(_ar1.gap_steps(np.array([0, 1, 3.1, 4]), 1.0), [1, 2, 1])


def test_fits_flag_time_gaps_and_ar1_time_is_opt_in():
    rng = np.random.default_rng(2)
    t = np.arange(300.0)
    y = rng.normal(size=300).cumsum() * 0.05 + rng.normal(size=300)
    assert "time_gaps" not in fit_poly(t, y, 0).flags
    y_gap = y.copy()
    y_gap[100:140] = np.nan
    default = fit_poly(t, y_gap, 0)
    timed = fit_poly(t, y_gap, 0, uncertainty="ar1_time")
    assert "time_gaps" in default.flags and "time_gaps" in timed.flags
    assert default.u["level"] != timed.u["level"]
    # Omitted timestamps are the same gap.
    keep = np.isfinite(y_gap)
    assert fit_poly(t[keep], y[keep], 0).u["level"] == default.u["level"]
    assert fit_poly(t[keep], y[keep], 0, uncertainty="ar1_time").u["level"] == timed.u["level"]
    # Without gaps, ar1_time is ar1.
    assert fit_poly(t, y, 1, uncertainty="ar1_time").u == fit_poly(t, y, 1).u
    with pytest.raises(ValueError, match="uncertainty must be one of"):
        fit_poly(t, y, 0, uncertainty="hac")


def test_residual_phi_uses_adjacent_pairs_only():
    e = np.array([1.0, 1.0, -1.0, -1.0, 1.0, 1.0])
    steps = np.array([1.0, 5.0, 1.0, 5.0, 1.0])  # pairs across the long steps are not lag 1
    r1 = (1 + 1 + 1) / 6 * 5 / 3
    assert residual_phi(e, 1, steps=steps) == pytest.approx(min(r1 + (1 + 3 * r1) / 6, 0.995))
    cov_t, _ = sandwich_cov(np.ones((6, 1)), e, "ar1_time", steps=steps)
    cov_c, _ = sandwich_cov(np.ones((6, 1)), e, "ar1", steps=steps)  # steps ignored
    assert cov_c[0, 0] == sandwich_cov(np.ones((6, 1)), e, "ar1")[0][0, 0] != cov_t[0, 0]


# -- provenance in reports ---------------------------------------------------------


def _record():
    t = np.arange(600.0)
    x = np.where(t < 300, 1.0, 1.0 + 2.0 * (1 - np.exp(-(t - 300) / 30.0)))
    x = x + np.random.default_rng(3).normal(0, 0.01, 600)
    idx = pd.date_range("2026-03-01 08:00", periods=600, freq="1s")
    return pd.DataFrame({"x": x}, index=idx)


def test_report_provenance_and_export(tmp_path):
    df = _record()
    det = rs.Detector([rs.Signal("x", "x")], [rs.DriftCriterion("x", 0.005)],
                      [rs.StepChange("x", 0.05, m=5)], timing=rs.Timing(block_s=10, window_s=60, max_gap_s=5))
    models = {"x": {"steady": 0, "transient": "first_order"}}
    rep = rs.batch.fit_report(det, df, models)
    p = rep.provenance
    assert p["kind"] == "fit_report" and p["rocksteady_version"] == rs.__version__
    assert p["time"] == {"unit": "s", "origin": "2026-03-01 08:00:00", "sample_period_s": 1.0}
    assert p["models"]["x"]["indeterminate"] == {"model": "first_order", "y0": None, "tau_min": None,
                                                 "tau_max": None, "uncertainty": "ar1"}
    assert p["parameter_order"]["x"]["steady"] == ["level"]
    assert p["nrmse"] == "mean" and p["error"] == "e = fit - data"
    assert p["detector"]["timing"]["block_samples"] == 10 and p["detector"]["timing"]["n_blocks"] == 6
    assert p["detector"]["validity_policy"]["max_gap_s"] == 5
    assert p["detector"]["criteria"] == [{"name": "x_drift", "type": "DriftCriterion", "signal": "x"}]
    assert p["segmentation"]["backfill"] is True
    assert rep.by_state.attrs["rocksteady"] is p and rep.by_interval.attrs["rocksteady"] is p
    assert "t_ref_s" in rep.by_interval.columns
    path = tmp_path / "report.json"
    rep.to_json(path)
    data = json.loads(path.read_text())
    assert data["provenance"]["models"] == json.loads(json.dumps(p["models"]))
    fits = data["fits"]["x"]
    assert len(fits) == len(rep.by_interval)
    # A fit can be evaluated from the export alone.
    i = next(k for k, f in enumerate(fits) if f["model"] == "first_order")
    row = rep.by_interval.iloc[i]
    t = np.arange(600.0)[int(row["start"]):int(row["stop"])]
    np.testing.assert_allclose(evaluate("first_order", fits[i]["evaluation"], t),
                               rep.fits["x"].fitted[int(row["start"]):int(row["stop"])], rtol=1e-14)


def test_compare_report_provenance():
    a = np.arange(10.0)
    rep = rs.report.compare_report(a, a + 1, [Segment(0, 10, State.STEADY)], names=("model", "data"), dt=1.0)
    assert rep.provenance["error"] == "e = model - data" and rep.provenance["kind"] == "compare_report"
    json.loads(rep.to_json())

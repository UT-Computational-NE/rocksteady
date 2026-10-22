"""Tests of the fit-model registry, the polynomial model and piecewise fits."""

import os
import sys

import numpy as np
import pytest

import rocksteady as rs
from rocksteady import State, registry, stats
from rocksteady.fit import (ar1_apply, fit_model, fit_poly, piecewise_fit, propagate_cov, resolve_model,
                            sandwich_cov, validate_models)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
from coverage_mc import ar1_noise  # noqa: E402

T0 = 1.7e9  # epoch-sized time origin


@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_poly_recovers_exact_polynomials_far_from_time_origin(order):
    t = T0 + np.arange(3600.0)
    c = t.mean()
    true = {"level": 3.0, "slope": 0.5, "curvature": 2e-4, "d3": 6e-8}
    y = np.full_like(t, 3.0)
    if order >= 1:
        y = y + 0.5 * (t - c)
    if order >= 2:
        y = y + 0.5 * 2e-4 * (t - c) ** 2
    if order >= 3:
        y = y + 6e-8 / 6 * (t - c) ** 3
    r = fit_poly(t, y, order)
    assert r.model == "poly" and r.label == f"poly {order}" and not r.flags
    assert list(r.params) == ["level", "slope", "curvature", "d3"][: order + 1]
    for k, v in r.params.items():
        assert v == pytest.approx(true[k], rel=1e-7)
    np.testing.assert_allclose(r.fitted, y, rtol=1e-10)


def test_residuals_are_orthogonal_to_the_basis():
    rng = np.random.default_rng(1)
    t = np.arange(500.0)
    r = fit_poly(t, np.sin(t / 50) + rng.normal(0, 0.1, 500), 3)
    x = (t - t.mean()) / (np.ptp(t) / 2)
    for k in range(4):
        assert abs(float((x**k) @ r.residuals)) < 1e-8


def test_order_reduction_nan_and_empty():
    r = fit_poly(np.arange(3.0), np.array([1.0, 2.0, 4.0]), 3)
    assert r.flags == {"order_reduced"} and r.options["order_used"] == 1
    assert np.isnan(r.params["curvature"]) and np.isnan(r.params["d3"]) and np.isfinite(r.params["slope"])
    y = np.array([1.0, np.nan, 3.0, 4.0, np.nan, 6.0])
    r = fit_poly(np.arange(6.0), y, 1)
    assert np.isnan(r.residuals[[1, 4]]).all() and np.isfinite(r.fitted).all()
    assert r.params["slope"] == pytest.approx(1.0)
    r = fit_poly(np.arange(3.0), np.full(3, np.nan), 1)
    assert r.flags == {"no_data"} and np.isnan(r.fitted).all() and r.cov is None
    with pytest.raises(ValueError):
        fit_poly(np.arange(3.0), np.ones(3), 1.5)


def test_iid_covariance_is_classical_and_ar1_close_for_white_noise():
    rng = np.random.default_rng(2)
    y = rng.normal(5.0, 2.0, 4000)
    t = np.arange(4000.0)
    r = fit_poly(t, y, 0, uncertainty="iid")
    assert r.u["level"] == pytest.approx(y.std(ddof=1) / np.sqrt(4000), rel=1e-12)
    r1 = fit_poly(t, y, 1, uncertainty="iid")
    stt = float((t - t.mean()) @ (t - t.mean()))
    s2 = float(r1.residuals @ r1.residuals) / (4000 - 2)
    assert r1.u["slope"] == pytest.approx(np.sqrt(s2 / stt), rel=1e-10)
    assert fit_poly(t, y, 0).u["level"] == pytest.approx(r.u["level"], rel=0.1)


def test_ar1_apply_matches_dense_matrix():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(60, 3))
    for phi in (0.0, 0.4, 0.95):
        dense = phi ** np.abs(np.subtract.outer(np.arange(60), np.arange(60)))
        np.testing.assert_allclose(ar1_apply(x, phi), dense @ x, rtol=1e-12)
        np.testing.assert_allclose(ar1_apply(x[:, 0], phi), dense @ x[:, 0], rtol=1e-12)


def test_line_uncertainty_matches_stats_slope_ar1():
    t = (np.arange(40) - 19.5) * 30.0
    y = 0.01 * t + ar1_noise(np.random.default_rng(4), 40, 0.5)
    b, u, _ = stats.slope_ar1(t, y)
    r = fit_poly(t, y, 1)
    assert r.params["slope"] == pytest.approx(b, rel=1e-12)
    assert r.u["slope"] == pytest.approx(u, rel=1e-10)


def test_sandwich_is_invariant_to_parameter_units():
    x = np.linspace(-1, 1, 100)
    e = np.sin(10 * x)
    rng = np.random.default_rng(6)
    ar = ar1_noise(rng, 100, 0.7)
    for method, resid in (("iid", e), ("ar1", ar)):
        ref = None
        for scale in (1.0, 1e-9, 1e9):
            cov, _ = sandwich_cov(np.column_stack((np.ones(100), scale * x)), resid, method=method)
            u = np.sqrt(cov[1, 1]) * scale
            ref = u if ref is None else ref
            assert u == pytest.approx(ref, rel=1e-8)
        if method == "iid":
            assert ref == pytest.approx(0.11947, rel=1e-4)  # the uncertainty at unit scale, which rescaling must keep


def test_sandwich_matches_dense_formula_when_full_rank():
    rng = np.random.default_rng(7)
    jac = np.column_stack((np.ones(80), rng.normal(size=80), rng.normal(size=80) ** 2))
    e = ar1_noise(rng, 80, 0.6)
    cov, phi = sandwich_cov(jac, e)
    r = phi ** np.abs(np.subtract.outer(np.arange(80), np.arange(80)))
    inv = np.linalg.inv(jac.T @ jac)
    tr_mr = 80 - np.trace(inv @ jac.T @ r @ jac)
    np.testing.assert_allclose(cov, float(e @ e) / tr_mr * inv @ jac.T @ r @ jac @ inv, rtol=1e-8)


def test_undetermined_directions_get_infinite_variance():
    x = np.linspace(-1, 1, 50)
    e = np.random.default_rng(8).normal(size=50)
    cov, _ = sandwich_cov(np.column_stack((np.ones(50), x, np.zeros(50))), e)
    assert np.isfinite(cov[0, 0]) and np.isfinite(cov[1, 1]) and np.isinf(cov[2, 2])
    cov, _ = sandwich_cov(np.column_stack((np.ones(50), 2 * np.ones(50))), e)  # collinear columns
    assert np.isinf(np.diag(cov)).all()
    out = propagate_cov(np.diag([1.0, np.inf]), np.array([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]]))
    assert out[0, 0] == 4.0 and np.isinf(out[1, 1]) and np.isinf(out[2, 2]) and np.isnan(out[0, 1])
    r = fit_poly(np.full(10, 5.0), np.arange(10.0), 1)  # no spread in time: slope undetermined
    assert "rank_deficient" in r.flags and np.isinf(r.u["slope"])


def test_sandwich_needs_degrees_of_freedom():
    assert sandwich_cov(np.ones((2, 2)), np.zeros(2))[0] is None
    with pytest.raises(ValueError):
        sandwich_cov(np.ones((5, 1)), np.ones(5), method="hac")


def test_model_specs():
    assert resolve_model(2) == ("poly", {"order": 2})
    assert resolve_model("poly") == ("poly", {})
    assert resolve_model({"model": "poly", "order": 1, "uncertainty": "iid"}) == ("poly", {"order": 1, "uncertainty": "iid"})
    by_state = {"steady": 0, "transient": {"model": "poly", "order": 3}}
    assert resolve_model(by_state, State.STEADY) == ("poly", {"order": 0})
    assert resolve_model(by_state, "indeterminate") == ("poly", {"order": 3})  # falls back to transient
    for bad, match in [(-1, ">= 0"), (True, "invalid"), ("spline", "unknown fit model"),
                       ({"steady": 0, "foo": 1}, "keys"), ({"steady": 0}, "no entry for state 'transient'")]:
        with pytest.raises((ValueError, KeyError), match=match):
            resolve_model(bad, "transient")
    with pytest.raises(ValueError, match="a state is needed"):
        resolve_model(by_state)
    validate_models({"power": by_state, "valve": 1})
    with pytest.raises(ValueError, match="model for 'power'"):
        validate_models({"power": {"steady": 0}})
    assert "poly" in registry.fit_models


def test_piecewise_fit_uses_the_model_for_each_state():
    t = np.arange(300.0)
    y = np.where(t < 100, 5.0, np.where(t < 200, 5.0 + 0.1 * (t - 100), 15.0))
    labels = ["steady"] * 100 + ["settling"] * 100 + ["steady"] * 100
    segs = rs.segments(labels)
    pw = piecewise_fit(t, y, segs, {"steady": 0, "transient": 1})
    assert [r.label for r in pw.results] == ["poly 0", "poly 1", "poly 0"]
    np.testing.assert_allclose(pw.fitted, y, atol=1e-9)
    for s, r in zip(segs, pw.results):
        alone = fit_model({"steady": 0, "transient": 1}, t[s.slice], y[s.slice], s.state)
        np.testing.assert_array_equal(r.fitted, alone.fitted)
    assert np.nanmax(np.abs(pw.residuals)) < 1e-9
    with pytest.raises(ValueError):
        piecewise_fit(t, y[:-1], segs, 0)


@pytest.mark.slow
@pytest.mark.parametrize("order", [0, 1])
@pytest.mark.parametrize("phi", [0.0, 0.5, 0.9])
def test_ar1_parameter_uncertainty_matches_spread(order, phi):
    """Over AR(1) noise, the reported u of the level (order 0) or slope (order 1) tracks the true spread."""
    rng = np.random.default_rng(5)
    t = np.arange(2000.0)
    name = ["level", "slope"][order]
    est, us = [], []
    for _ in range(300):
        r = fit_poly(t, ar1_noise(rng, 2000, phi), order)
        est.append(r.params[name])
        us.append(r.u[name])
    assert 0.85 < np.median(us) / np.std(est) < 1.15

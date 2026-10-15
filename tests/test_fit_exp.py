"""Tests of the exponential fit models (first_order, exp)."""

import os
import sys

import numpy as np
import pytest

import rocksteady as rs
from rocksteady import registry
from rocksteady.fit import fit_exp, fit_first_order, piecewise_fit

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
from coverage_mc import ar1_noise  # noqa: E402

T0 = 1.7e9


@pytest.mark.parametrize("y0, y_inf, tau", [(10.0, 20.0, 120.0), (8.0, 5.0, 200.0), (-3.0, 4.0, 30.0)])
def test_first_order_recovers_rise_and_decay(y0, y_inf, tau):
    t = T0 + np.arange(1000.0)
    y = y_inf + (y0 - y_inf) * np.exp(-(t - T0) / tau)
    r = fit_first_order(t, y)
    assert r.model == "first_order" and not r.flags
    for k, v in {"y0": y0, "y_inf": y_inf, "tau": tau, "lambda": 1 / tau}.items():
        assert r.params[k] == pytest.approx(v, rel=1e-6)
    np.testing.assert_allclose(r.fitted, y, atol=1e-8)


def test_first_order_with_fixed_start_is_A_one_minus_exp():
    t = np.arange(800.0)
    r = fit_first_order(t, 7.0 * (1 - np.exp(-0.01 * t)), y0=0.0)
    assert r.params["y0"] == 0.0 and r.u["y0"] == 0.0
    assert r.params["y_inf"] == pytest.approx(7.0, rel=1e-6) and r.params["lambda"] == pytest.approx(0.01, rel=1e-6)


@pytest.mark.parametrize("c, A, T", [(1.0, 0.5, 40.0), (2.0, 50.0, -80.0), (0.0, 3.0, 150.0)])
def test_exp_recovers_growth_and_decay(c, A, T):
    t = T0 + np.arange(300.0)
    y = c + A * np.exp((t - T0) / T)
    r = fit_exp(t, y)
    assert not r.flags
    for k, v in {"c": c, "A": A, "T": T}.items():
        assert r.params[k] == pytest.approx(v, rel=1e-5, abs=1e-8)


def test_flags_for_unresolvable_time_constants():
    t = np.arange(50.0)
    rng = np.random.default_rng(0)
    # Interval far shorter than tau: looks like a line.
    r = fit_first_order(t, 10 * (1 - np.exp(-t / 2000.0)) + rng.normal(0, 0.01, 50))
    assert "tau_unidentifiable" in r.flags
    # Time constant under five sample spacings: a step, not a curve.
    r = fit_first_order(t, 10 * (1 - np.exp(-t / 1.5)) + rng.normal(0, 0.01, 50))
    assert "tau_unidentifiable" in r.flags
    # Pure noise: no exponential to find.
    r = fit_exp(np.arange(200.0), rng.normal(0, 1, 200))
    assert r.flags & {"tau_unidentifiable", "tau_at_bound"}
    r = fit_first_order(np.arange(3.0), np.ones(3))
    assert r.flags == {"too_few_samples"} and np.isnan(r.params["tau"])


def test_nan_samples_are_ignored():
    t = np.arange(500.0)
    y = 20 + (10 - 20) * np.exp(-t / 80)
    y[[3, 100, 250]] = np.nan
    r = fit_first_order(t, y)
    assert r.params["tau"] == pytest.approx(80, rel=1e-6)
    assert np.isnan(r.residuals[[3, 100, 250]]).all() and np.isfinite(r.fitted).all()


def test_models_work_through_specs_and_piecewise_fits():
    assert {"poly", "first_order", "exp"} <= set(registry.fit_models)
    t = np.arange(900.0)
    y = np.where(t < 300, 10.0, 20 + (10 - 20) * np.exp(-(t - 300) / 60))
    segs = rs.segments(["steady"] * 300 + ["settling"] * 600)
    pw = piecewise_fit(t, y, segs, {"steady": 0, "transient": {"model": "first_order", "tau_max": 1e4}})
    assert [r.label for r in pw.results] == ["poly 0", "first_order"]
    assert pw.results[1].params["tau"] == pytest.approx(60, rel=1e-6)
    np.testing.assert_allclose(pw.fitted, y, atol=1e-8)


@pytest.mark.slow
@pytest.mark.parametrize("phi", [0.0, 0.5, 0.9])
def test_time_constant_uncertainty_covers_truth(phi):
    """tau +/- 2 u(tau) and T +/- 2 u(T) cover the truth close to the nominal 95.4 % under AR(1) noise."""
    rng = np.random.default_rng(7)
    t = np.arange(600.0)
    sd = 0.5 * np.sqrt(1 - phi**2)
    reps, hit_tau, hit_T = 300, 0, 0
    for _ in range(reps):
        r = fit_first_order(t, 10 * (1 - np.exp(-t / 100)) + sd * ar1_noise(rng, 600, phi))
        hit_tau += abs(r.params["tau"] - 100) <= 2 * r.u["tau"]
        r = fit_exp(t[:300], 0.2 * np.exp(t[:300] / 60) + sd * ar1_noise(rng, 300, phi))
        hit_T += abs(r.params["T"] - 60) <= 2 * r.u["T"]
    assert 0.88 <= hit_tau / reps <= 0.99
    assert 0.88 <= hit_T / reps <= 0.99

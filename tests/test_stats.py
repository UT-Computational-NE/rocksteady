"""Tests of the statistics: Student t, slope methods, uncertainty of a mean."""

import os
import sys

import numpy as np
import pytest

import rocksteady as rs
from rocksteady import registry, stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
from coverage_mc import ar1_noise, false_pass_rate  # noqa: E402


@pytest.mark.parametrize("p, nu, want", [(0.975, 1, 12.7062), (0.975, 2, 4.30265), (0.975, 3, 3.18245),
                                         (0.975, 10, 2.22814), (0.95, 30, 1.69726), (0.975, 120, 1.97993)])
def test_t_quantile(p, nu, want):
    assert stats.t_quantile(p, nu) == pytest.approx(want, abs=1e-4)


def test_slope_methods_are_registered_and_default_is_ar1():
    assert {"ar1", "nw"} <= set(registry.slope_methods)
    assert rs.DriftCriterion("x", 1.0).method == stats.DEFAULT_SLOPE_METHOD == "ar1"


def test_slope_ar1_on_white_noise_and_exact_line():
    t = (np.arange(40) - 19.5) * 30.0
    b, u, k = stats.slope_ar1(t, 2.0 + 1e-3 * t)
    assert b == pytest.approx(1e-3) and u == 0.0 and k == pytest.approx(stats.t_quantile(0.975, 38))
    rng = np.random.default_rng(3)
    b, u, k = stats.slope_ar1(t, rng.normal(size=40))
    sd_true = 1.0 / np.sqrt(float((t - t.mean()) @ (t - t.mean())))
    assert 0.6 < u / sd_true < 1.5 and k >= stats.t_quantile(0.975, 38)
    assert np.isnan(stats.slope_ar1(t[:3], np.ones(3))[1])


def test_drift_criterion_method_options_and_explicit_k():
    t = (np.arange(40) - 19.5) * 30.0
    y = np.random.default_rng(4).normal(size=40)
    legacy = rs.DriftCriterion("x", 1.0, method="nw", k=2.0).evaluate(t, y)
    b, u = stats.hac_slope(t, y)
    assert (legacy.statistic, legacy.uncertainty, legacy.k) == (abs(b), u, 2.0)
    narrow = rs.DriftCriterion("x", 1.0, method="nw", method_options={"max_lag_frac": 0.0}).evaluate(t, y)
    assert narrow.k == 2.0
    custom = rs.DriftCriterion("x", 1.0, method=lambda t, y: (0.1, 0.01, 3.0)).evaluate(t, y)
    assert (custom.statistic, custom.uncertainty, custom.k) == (0.1, 0.01, 3.0)
    with pytest.raises(KeyError):
        rs.DriftCriterion("x", 1.0, method="nope")


def test_mean_with_uncertainty_type_b():
    x = np.random.default_rng(5).normal(10.0, 1.0, 2000)
    m = stats.mean_with_uncertainty(x, 1.0)
    assert m["u_combined"] == pytest.approx(m["u_mean"])
    mb = stats.mean_with_uncertainty(x, 1.0, u_type_b=0.05)
    assert mb["u_combined"] == pytest.approx(np.hypot(m["u_mean"], 0.05))
    assert m["u_mean"] == pytest.approx(1.0 / np.sqrt(2000), rel=0.15)


@pytest.mark.slow
@pytest.mark.parametrize("phi", [0.0, 0.5, 0.8])
def test_ar1_false_pass_rate_holds_target(phi):
    """At 40 blocks the calibrated method stays within twice its nominal 2.5 %."""
    assert false_pass_rate("ar1", 40, phi, reps=3000, seed=0) <= 0.05


@pytest.mark.slow
def test_nw_false_pass_rate_is_too_high_when_autocorrelated():
    """Documents why "nw" is not the default."""
    assert false_pass_rate("nw", 40, 0.5, reps=3000, seed=0) > 0.08


@pytest.mark.slow
@pytest.mark.parametrize("phi", [0.0, 0.5, 0.9])
def test_mean_uncertainty_matches_truth_for_ar1(phi):
    n, reps = 1800, 200
    rng = np.random.default_rng(6)
    true_var = (n - 2 * phi - n * phi**2 + 2 * phi ** (n + 1)) / (n**2 * (1 - phi) ** 2) / (1 - phi**2)  # Zhang eq. 7
    u = [stats.mean_with_uncertainty(ar1_noise(rng, n, phi), 1.0)["u_mean"] for _ in range(reps)]
    assert 0.9 < np.median(u) / np.sqrt(true_var) < 1.1

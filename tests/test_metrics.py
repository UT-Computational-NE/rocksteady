"""Tests of the pooled autocorrelation time and the interval/state statistics."""

import os
import sys

import numpy as np
import pytest

import rocksteady as rs
from rocksteady import stats
from rocksteady.fit import fit_first_order, fit_poly
from rocksteady.metrics import error_stats, finite_pieces, signal_stats, stats_by_segment, stats_by_state

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
from coverage_mc import ar1_noise  # noqa: E402


def _tau_int_reference(x, dt, max_lag_frac=0.25):
    """The rocksteady 0.2 implementation of tau_int (direct lag loop)."""
    n = len(x)
    _, _, r = stats.linear_fit(np.arange(n, dtype=float), np.asarray(x, dtype=float))
    var = float((r * r).mean())
    if var <= 0:
        return dt
    s = 0.0
    for k in range(1, max(int(n * max_lag_frac), 1) + 1):
        rho = float((r[:-k] * r[k:]).mean()) / var
        if rho <= 0:
            break
        s += rho
    return dt * max(1.0, 1.0 + 2.0 * s)


@pytest.mark.parametrize("n, phi", [(50, 0.0), (500, 0.7), (5000, 0.95)])  # the last uses the FFT path
def test_tau_int_unchanged_and_equal_to_acf_tau(n, phi):
    x = ar1_noise(np.random.default_rng(n), n, phi)
    assert stats.tau_int(x, 2.0) == pytest.approx(_tau_int_reference(x, 2.0), rel=1e-10)
    _, _, r = stats.linear_fit(np.arange(n, dtype=float), x)
    assert stats.tau_int(x, 2.0) == stats.acf_tau([r], 2.0)


def test_acf_tau_pools_within_pieces_only():
    p = np.sin(np.linspace(0, 3 * np.pi, 200))
    assert stats.acf_tau([p, p], 1.0) == pytest.approx(stats.acf_tau([p], 1.0), rel=1e-12)
    assert stats.acf_tau([p, -p], 1.0) == pytest.approx(stats.acf_tau([p], 1.0), rel=1e-12)
    assert stats.acf_tau([np.concatenate((p, -p))], 1.0) != pytest.approx(stats.acf_tau([p], 1.0), rel=1e-3)
    assert np.isnan(stats.acf_tau([], 1.0)) and stats.acf_tau([np.zeros(10)], 3.0) == 3.0


def test_pooled_uncertainty_weights_lags_by_pairs_within_pieces():
    # 10,000 independent values, each repeated twice: the mean's uncertainty is 1/sqrt(10,000).
    k = 10_000
    x = np.random.default_rng(20).normal(size=k)
    e = np.repeat(x, 2)
    r = error_stats(e, np.zeros_like(e), dt=1.0, pieces=[(2 * i, 2 * i + 2) for i in range(k)])
    assert r.tau_int_s == pytest.approx(2.0) and r.n_eff == pytest.approx(k)
    assert r.u_bias == pytest.approx(x.std(ddof=1) / np.sqrt(k), rel=0.01)
    # Many independent singletons plus one strongly correlated piece, all with unit variance: the
    # correlation applies only to the 100 samples that have partners.
    phi = 0.95
    ratios = []
    for seed in range(40):
        rng = np.random.default_rng(seed)
        a = np.concatenate((rng.normal(size=10_000), ar1_noise(rng, 100, phi) * np.sqrt(1 - phi**2)))
        pieces = [(i, i + 1) for i in range(10_000)] + [(10_000, 10_100)]
        r = error_stats(a, np.zeros_like(a), dt=1.0, pieces=pieces)
        true_var = (10_000 + 100**2 * _var_mean_ar1(100, phi) * (1 - phi**2)) / 10_100**2
        ratios.append(r.u_bias / np.sqrt(true_var))
    assert 0.85 < np.median(ratios) < 1.15
    # The unweighted time constant is unchanged.
    assert stats.acf_tau([e[2 * i: 2 * i + 2] for i in range(k)], 1.0) == pytest.approx(3.0)


def test_finite_pieces():
    m = np.array([1, 1, 0, 1, 1, 1, 0, 0, 1], dtype=bool)
    assert finite_pieces(m) == [(0, 2), (3, 6), (8, 9)]
    assert finite_pieces(m, [(0, 4), (4, 9)]) == [(0, 2), (3, 4), (4, 6), (8, 9)]


def test_error_stats_closed_form():
    b = np.linspace(10.0, 20.0, 101)
    es = error_stats(b + 2.0, b, dt=0.5)
    assert (es.n, es.duration_s) == (101, 50.5)
    assert es.bias == pytest.approx(2.0) and es.std == pytest.approx(0.0, abs=1e-12)
    assert es.mae == pytest.approx(2.0) and es.rmse == pytest.approx(2.0) and es.max_abs == pytest.approx(2.0)
    assert es.r == pytest.approx(1.0) and es.nrmse == pytest.approx(2.0 / 15.0)
    assert error_stats(b + 2.0, b, dt=0.5, nrmse="range").nrmse == pytest.approx(0.2)
    assert error_stats(b + 2.0, b, dt=0.5, nrmse="std").nrmse == pytest.approx(2.0 / b.std(ddof=1))
    assert np.isnan(error_stats(b + 2.0, b, dt=0.5, nrmse="none").nrmse)
    e = np.array([1.0, -1.0, 3.0, -3.0])
    es = error_stats(e, np.zeros(4), dt=1.0, nrmse="none")
    assert es.bias == 0.0 and es.mae == 2.0 and es.rmse == pytest.approx(np.sqrt(5.0)) and es.max_abs == 3.0
    assert np.isnan(es.r)  # b is constant
    with pytest.raises(ValueError):
        error_stats(e, np.zeros(4), dt=1.0, nrmse="max")


def test_error_stats_ignores_nan_and_honours_pieces():
    a = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 100.0])
    b = np.zeros(6)
    es = error_stats(a, b, dt=1.0, pieces=[(0, 5)])
    assert es.n == 4 and es.bias == pytest.approx(3.0) and es.max_abs == 5.0
    assert error_stats(a, b, dt=1.0, pieces=[]).n == 0


def test_u_bias_is_at_least_iid_and_grows_with_autocorrelation():
    rng = np.random.default_rng(1)
    white = rng.normal(size=4000)
    red = ar1_noise(rng, 4000, 0.9)
    for e in (white, red):
        es = error_stats(e, np.zeros_like(e), dt=1.0)
        assert es.u_bias >= es.std / np.sqrt(es.n) * 0.999
    es = error_stats(red, np.zeros_like(red), dt=1.0)
    assert es.n_eff < es.n / 10 and es.tau_int_s > 10


def test_signal_stats_about_the_fit():
    t = np.arange(1000.0)
    rng = np.random.default_rng(2)
    y = 20 + (10 - 20) * np.exp(-t / 80) + rng.normal(0, 0.1, 1000)
    ss = signal_stats(y, fit_first_order(t, y), dt=1.0)
    assert ss.model == "first_order" and ss.params["tau"] == pytest.approx(80, rel=0.05)
    assert ss.std == pytest.approx(0.1, rel=0.1) and ss.tau_int_s < 3  # the fit removed the trend
    const = signal_stats(y, fit_poly(t, y, 0), dt=1.0)
    assert const.tau_int_s > 50 and const.u_mean > 10 * ss.u_mean  # a constant leaves the trend in the "noise"
    assert ss.mean == pytest.approx(y.mean())
    assert signal_stats(np.full(5, np.nan), fit_poly(np.arange(5.0), np.full(5, np.nan), 0), dt=1.0).n == 0


def test_stats_by_state_and_segment():
    labels = ["indeterminate"] * 50 + ["steady"] * 200 + ["settling"] * 100 + ["steady"] * 150
    segs = rs.segments(labels)
    rng = np.random.default_rng(3)
    b = rng.normal(size=500)
    a = b + 1.0
    table = stats_by_state(a, b, segs, dt=1.0)
    assert list(table.index) == ["all", "steady", "transient", "indeterminate"]
    assert table.loc["all", "n"] == 500
    assert table.loc[["steady", "transient", "indeterminate"], "n"].sum() == 500
    assert table.loc["steady", "n"] == 350 and (table["bias"] - 1.0).abs().max() < 1e-12
    only = stats_by_state(a, b, rs.segments(["steady"] * 10), dt=1.0)
    assert only.loc["transient", "n"] == 0 and np.isnan(only.loc["transient", "bias"])
    per = stats_by_segment(a, b, segs, dt=1.0)
    assert list(per["state"]) == ["indeterminate", "steady", "transient", "steady"]
    assert list(per["n"]) == [50, 200, 100, 150]


def _var_mean_ar1(n, phi):
    """Variance of the mean of n AR(1) samples with unit innovations (Zhang 2006, eq. 7)."""
    return (n - 2 * phi - n * phi**2 + 2 * phi ** (n + 1)) / (n**2 * (1 - phi) ** 2) / (1 - phi**2)


@pytest.mark.slow
@pytest.mark.parametrize("phi", [0.0, 0.5, 0.9])
def test_u_bias_and_u_mean_match_true_uncertainty(phi):
    rng = np.random.default_rng(4)
    k, m = 5, 400
    pieces = [(i * m, (i + 1) * m) for i in range(k)]
    ub, um = [], []
    for _ in range(300):
        e = np.concatenate([ar1_noise(rng, m, phi) for _ in range(k)]) + 3.0
        ub.append(error_stats(e, np.zeros_like(e), dt=1.0, pieces=pieces).u_bias)
        y = ar1_noise(rng, 2000, phi)
        um.append(signal_stats(y, fit_poly(np.arange(2000.0), y, 0), dt=1.0).u_mean)
    assert 0.9 < np.median(ub) / np.sqrt(_var_mean_ar1(m, phi) / k) < 1.1
    assert 0.9 < np.median(um) / np.sqrt(_var_mean_ar1(2000, phi)) < 1.1

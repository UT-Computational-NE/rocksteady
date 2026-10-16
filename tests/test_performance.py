"""The O(n) numerics agree with the dense reference algorithms they replaced (#25)."""

import math

import numpy as np
import pytest

from rocksteady import _ar1, fit, stats


def _slope_ar1_dense(t, y, confidence=0.975, phi_max=0.98):
    """The 0.3.0 implementation, with a dense n x n correlation matrix."""
    n = len(y)
    tc = t - t.mean()
    stt = float(tc @ tc)
    _, b, e = stats.linear_fit(t, y)
    ee = float(e @ e)
    if ee <= 0:
        return b, 0.0, stats.t_quantile(confidence, n - 2)
    r1 = float(e[1:] @ e[:-1]) / ee
    phi = min(max(r1 + (2.0 + 4.0 * r1) / n, 0.0), phi_max)
    corr = phi ** np.abs(np.subtract.outer(np.arange(n), np.arange(n))).astype(float)
    x = np.column_stack((np.ones(n), tc))
    tr_mr = n - float(np.trace(np.linalg.solve(x.T @ x, x.T @ corr @ x)))
    c = tc / stt
    u = math.sqrt(max(ee / tr_mr * float(c @ corr @ c), 0.0))
    nu = max(int(n * (1.0 - phi) / (1.0 + phi)) - 2, 1)
    return b, u, stats.t_quantile(confidence, nu)


def _ar1_apply_numpy(x, phi):
    """The 0.3.0 ``fit.ar1_apply`` (numpy row recursion)."""
    x = np.asarray(x, dtype=float)
    if phi == 0.0:
        return x.copy()
    f = np.empty_like(x)
    b = np.empty_like(x)
    f[0] = x[0]
    for i in range(1, len(x)):
        f[i] = x[i] + phi * f[i - 1]
    b[-1] = x[-1]
    for i in range(len(x) - 2, -1, -1):
        b[i] = x[i] + phi * b[i + 1]
    return f + b - x


def _acf_tau_reference(pieces, dt, max_lag_frac=0.25, pair_weighted=False):
    """The 0.3.0 ``acf_tau``: every piece padded to the longest piece's lag cap."""
    pieces = [np.asarray(p, dtype=float) for p in pieces if len(p) > 0]
    lens = np.array([len(p) for p in pieces])
    max_lag = max(int(lens.max() * max_lag_frac), 1)
    sums = np.zeros(max_lag + 1)
    counts = np.zeros(max_lag + 1)
    ks = np.arange(max_lag + 1)
    for p in pieces:
        n = len(p)
        s = np.zeros(max_lag + 1)  # padded with zeros to max_lag (the lag sums themselves are unchanged)
        local = stats._lag_sums(p, max_lag)
        s[: len(local)] = local
        sums += s
        counts += np.maximum(n - ks, 0)
    var = sums[0] / counts[0]
    total = 0.0
    for k in range(1, max_lag + 1):
        if counts[k] == 0:
            break
        rho = sums[k] / counts[k] / var
        if rho <= 0:
            break
        total += rho * (counts[k] / counts[0] if pair_weighted else 1.0)
    return dt * max(1.0, 1.0 + 2.0 * total)


@pytest.mark.parametrize("n", [4, 10, 40, 90, 300, 1000])
@pytest.mark.parametrize("ar", [0.0, 0.5, 0.9, 0.99])
def test_slope_ar1_matches_the_dense_algorithm(n, ar):
    rng = np.random.default_rng(n + int(100 * ar))
    e = np.zeros(n)
    w = rng.normal(size=n)
    for i in range(n):
        e[i] = (ar * e[i - 1] if i else 0.0) + w[i]
    t = 30.0 * np.arange(n) + 1e5  # block times far from zero
    y = 0.01 * t + e
    b, u, k = stats.slope_ar1(t, y)
    b0, u0, k0 = _slope_ar1_dense(t, y)
    assert b == b0 and k == k0
    np.testing.assert_allclose(u, u0, rtol=1e-11)


def test_no_unbounded_cache():
    assert not hasattr(stats, "_LAGS") and not hasattr(stats, "_lag_matrix")


@pytest.mark.parametrize("phi", [0.0, 0.3, 0.9, 0.995])
def test_ar1_apply_is_unchanged(phi):
    rng = np.random.default_rng(5)
    for shape in [(1,), (7,), (500,), (500, 3), (2, 2)]:
        x = rng.normal(size=shape)
        np.testing.assert_array_equal(_ar1.ar1_apply(x, phi), _ar1_apply_numpy(x, phi))
    assert fit.ar1_apply is _ar1.ar1_apply  # still importable from rocksteady.fit
    assert _ar1.ar1_apply(np.zeros((0, 2)), 0.5).shape == (0, 2)


@pytest.mark.parametrize("pair_weighted", [False, True])
def test_acf_tau_is_unchanged_on_fragmented_pieces(pair_weighted):
    rng = np.random.default_rng(6)
    long_piece = np.convolve(rng.normal(size=5000), np.ones(8) / 8, mode="same")
    pieces = [long_piece] + [rng.normal(size=int(m)) for m in rng.integers(1, 40, size=300)]
    for ps in (pieces, pieces[1:], pieces[::-1], [long_piece]):
        want = _acf_tau_reference(ps, 0.5, pair_weighted=pair_weighted)
        assert stats.acf_tau(ps, 0.5, pair_weighted=pair_weighted) == want

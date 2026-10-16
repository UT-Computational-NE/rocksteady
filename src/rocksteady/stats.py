"""
Statistics shared by the criteria and by post-acceptance reporting.

Slope uncertainty
-----------------
A drift test needs the slope of a window of block means and an uncertainty
that survives autocorrelation. Slope methods are registered in
``rocksteady.registry.slope_methods``. Each is a callable
``(t, y, **options) -> (b, u_b, k)`` where ``b + k * u_b`` is a one-sided
upper confidence bound on the slope (the criterion uses ``|b| + k u_b``).

- ``"ar1"`` (default): OLS slope with an AR(1) error model. The lag-1
  autocorrelation of the residuals is bias-corrected for a regression on a
  trend, ``phi + (2 + 4 phi) / n``, then clipped to [0, 0.98]. ``u_b`` is
  exact for AR(1) errors with that ``phi``, the error variance is
  estimated without bias given ``phi``, and ``k`` is the Student-t
  quantile with ``n (1 - phi) / (1 + phi) - 2`` degrees of freedom (the
  effective sample size), at 97.5 % one-sided by default.
- ``"nw"``: Newey-West (Bartlett) standard error with a data-dependent
  bandwidth (first non-positive lag of the residual autocorrelation,
  capped at a quarter of the window) and ``k = 2``. This was the method
  of rocksteady 0.1; it is kept for reproducing earlier results.

Monte Carlo false-pass rates of the equivalence test at the boundary (true
slope equal to the limit), AR(1) block-mean noise, nominal 2.5 % for
``"ar1"`` (2.3 % for ``"nw"`` with k = 2); ``tools/coverage_mc.py``, 3000
replications, seed 0:

    n blocks         10                    40                    90
    phi        0   .5   .8   .9      0   .5   .8   .9      0   .5   .8   .9
    ar1       1.7  5.1  7.7  9.3    1.9  3.9  4.6  4.8    2.4  3.0  3.3  4.1
    nw        7.2 16.0 24.6 27.9    3.6 10.6 17.7 23.5    2.8  7.3 12.9 18.2

``"nw"`` understates the slope uncertainty for autocorrelated windows, so
it certifies drift limits the data do not support. ``"ar1"`` holds the
target (at most twice nominal) from about 40 blocks with ``phi <= 0.8``.
With 10 blocks no method here is reliable. Use longer windows, or longer
blocks so that block means are less correlated.

The price of ``"ar1"`` is power. When the residual autocorrelation is
high the effective sample size is small and ``k`` grows (to 12.7 at one
degree of freedom), so a window can certify only a limit well above its
apparent drift. That is the honest answer for such data: a 20-minute
window of block means with lag-1 autocorrelation 0.9 holds about two
independent samples.

Uncertainty of a mean
---------------------
``tau_int`` and ``mean_with_uncertainty`` give the Type A uncertainty of
the mean over an accepted interval through an effective sample count; see
their docstrings for the estimator choices.
"""

import math
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np

from . import _validate as _v
from . import registry
from ._ar1 import ar1_quad

DEFAULT_SLOPE_METHOD = "ar1"


def block_means(x: np.ndarray, k: int) -> np.ndarray:
    """Non-overlapping means of consecutive groups of ``k`` samples."""
    n = (len(x) // k) * k
    return x[:n].reshape(-1, k).mean(axis=1)


def first_nonpositive_lag(resid: np.ndarray, cap: int) -> int:
    """Number of leading lags with positive autocorrelation, capped."""
    r = resid - resid.mean()
    var = float((r * r).mean())
    if var <= 0:
        return 0
    lag = 0
    for ell in range(1, min(cap, len(r) - 1) + 1):
        rho = float((r[:-ell] * r[ell:]).mean()) / var
        if rho <= 0:
            break
        lag = ell
    return lag


def linear_fit(t: np.ndarray, y: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """OLS line ``y = a + b t``. Returns (a, b, residuals)."""
    tc = t - t.mean()
    stt = float(tc @ tc)
    if stt <= 0:
        return float(y.mean()), 0.0, y - y.mean()
    b = float(tc @ (y - y.mean())) / stt
    a = float(y.mean() - b * t.mean())
    return a, b, y - (a + b * t)


# ---------------------------------------------------------------------------
# Student t (exact for integer degrees of freedom; no scipy needed)
# ---------------------------------------------------------------------------


def _t_cdf(x: float, nu: int) -> float:
    """Student-t CDF for integer ``nu`` (Abramowitz and Stegun 26.7.3-4)."""
    th = math.atan(abs(x) / math.sqrt(nu))
    c2, s = math.cos(th) ** 2, math.sin(th)
    if nu % 2:
        if nu == 1:
            a = 2.0 * th / math.pi
        else:
            term = tot = math.cos(th)
            for j in range(3, nu - 1, 2):
                term *= (j - 1) / j * c2
                tot += term
            a = 2.0 / math.pi * (th + s * tot)
    else:
        term = tot = 1.0
        for j in range(2, nu - 1, 2):
            term *= (j - 1) / j * c2
            tot += term
        a = s * tot
    return 0.5 + 0.5 * a if x >= 0 else 0.5 - 0.5 * a


@lru_cache(maxsize=4096)
def t_quantile(p: float, nu: int) -> float:
    """Quantile of Student's t with integer ``nu`` degrees of freedom, for 0.5 < p < 1."""
    if not 0.5 < p < 1.0 or nu < 1:
        raise ValueError("t_quantile needs 0.5 < p < 1 and nu >= 1")
    lo, hi = 0.0, 1.0
    while _t_cdf(hi, nu) < p:
        hi *= 2.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _t_cdf(mid, nu) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12 * hi:
            break
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Slope uncertainty methods
# ---------------------------------------------------------------------------


def hac_slope(t: np.ndarray, y: np.ndarray, max_lag_frac: float = 0.25) -> Tuple[float, float]:
    """
    OLS slope with a Newey-West (Bartlett) standard error.

    Returns (b, u_b) in y-units per t-unit. The bandwidth is the first
    non-positive lag of the residual autocorrelation, capped at
    ``max_lag_frac`` of the sample count.
    """
    n = len(y)
    tc = t - t.mean()
    stt = float(tc @ tc)
    if stt <= 0 or n < 4:
        return np.nan, np.nan
    _, b, e = linear_fit(t, y)
    g = tc * e
    lag = first_nonpositive_lag(e, max(int(n * max_lag_frac), 1))
    s = float(g @ g)
    for ell in range(1, lag + 1):
        s += 2.0 * (1.0 - ell / (lag + 1.0)) * float(g[ell:] @ g[:-ell])
    return b, float(np.sqrt(max(s / (stt * stt), 0.0)))


@registry.slope_methods.register("nw")
def slope_nw(t: np.ndarray, y: np.ndarray, max_lag_frac: float = 0.25, k: float = 2.0) -> Tuple[float, float, float]:
    """Newey-West slope uncertainty (``hac_slope``) with a fixed ``k``. The rocksteady 0.1 method."""
    b, u = hac_slope(t, y, max_lag_frac)
    return b, u, k


def _check_nw_options(max_lag_frac: float = 0.25, k: float = 2.0) -> None:
    if not 0 <= _v.finite(max_lag_frac, "max_lag_frac") <= 1:  # the bandwidth is at least one lag
        raise ValueError(f"max_lag_frac must be in [0, 1], got {max_lag_frac!r}")
    _v.finite(k, "k", nonnegative=True)


def _check_ar1_options(confidence: float = 0.975, phi_max: float = 0.98) -> None:
    if not 0.5 < _v.finite(confidence, "confidence") < 1:
        raise ValueError(f"confidence must be in (0.5, 1), got {confidence!r}")
    if not 0 <= _v.finite(phi_max, "phi_max") < 1:
        raise ValueError(f"phi_max must be in [0, 1), got {phi_max!r}")


slope_nw.validate_options = _check_nw_options


@registry.slope_methods.register("ar1")
def slope_ar1(t: np.ndarray, y: np.ndarray, confidence: float = 0.975,
              phi_max: float = 0.98) -> Tuple[float, float, float]:
    """
    OLS slope with an AR(1) error model (see the module docstring).

    Returns (b, u_b, k) with ``k`` the one-sided Student-t quantile at
    ``confidence`` on the effective degrees of freedom.
    """
    n = len(y)
    tc = t - t.mean()
    stt = float(tc @ tc)
    if stt <= 0 or n < 4:
        return np.nan, np.nan, np.nan
    _, b, e = linear_fit(t, y)
    ee = float(e @ e)
    if ee <= 0:
        return b, 0.0, t_quantile(confidence, n - 2)
    r1 = float(e[1:] @ e[:-1]) / ee
    phi = min(max(r1 + (2.0 + 4.0 * r1) / n, 0.0), phi_max)  # bias correction for a trend regression
    x = np.column_stack((np.ones(n), tc))
    xrx = ar1_quad(x, phi)  # X'RX in O(n), without forming R
    # E[e'e] = sigma^2 tr(M R) for residual maker M; tr(M R) = n - tr((X'X)^-1 X'RX).
    tr_mr = n - float(np.trace(np.linalg.solve(x.T @ x, xrx)))
    # Var(b) = sigma^2 c'Rc with c = tc / stt, and tc'R tc = (X'RX)_11.
    u = math.sqrt(max(ee / tr_mr * float(xrx[1, 1]) / (stt * stt), 0.0))
    nu = max(int(n * (1.0 - phi) / (1.0 + phi)) - 2, 1)
    return b, u, t_quantile(confidence, nu)


slope_ar1.validate_options = _check_ar1_options


# ---------------------------------------------------------------------------
# Uncertainty of a mean
# ---------------------------------------------------------------------------


def tau_int(x: np.ndarray, dt: float, max_lag_frac: float = 0.25) -> float:
    """
    Integrated autocorrelation time (same units as ``dt``) of the detrended series.

    ``tau_int = dt * (1 + 2 sum_k rho(k))``, floored at ``dt``, with:

    - the series detrended by an OLS line first, so that a residual drift
      within the accepted interval does not inflate the correlation time;
    - ``rho(k)`` the lag-normalised estimate, ``mean(r_i r_{i+k}) / mean(r_i^2)``;
    - the sum unweighted and stopped before the first non-positive
      ``rho(k)``, or at ``max_lag_frac`` of the record.

    This is not the estimator of Zhang (2006, Metrologia 43 S276), which
    uses the standard ``rho`` estimate with ``(n - k)/n`` weights and cuts
    off at the largest lag whose ``rho`` is significant. In Monte Carlo
    checks on AR(1) data that cutoff often picks up spurious lags (median
    ~360 for white noise with n = 1800) and underestimates the uncertainty
    by about 25 %. The estimator here was within 3 % for ``phi <= 0.9``.
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return dt
    _, _, r = linear_fit(np.arange(len(x), dtype=float), x)
    return acf_tau([r], dt, max_lag_frac)


def _lag_sums(r: np.ndarray, max_lag: int) -> np.ndarray:
    """``sum_i r_i r_(i+k)`` for ``k = 0 .. min(max_lag, n - 1)`` within one piece of length ``n``."""
    n = len(r)
    lags = min(max_lag, n - 1)
    out = np.zeros(lags + 1)
    if n * (lags + 1) <= 200_000:
        out[: lags + 1] = [float(r[: n - k] @ r[k:]) for k in range(lags + 1)]
    else:  # FFT autocorrelation, O(n log n)
        m = 1 << int(np.ceil(np.log2(2 * n)))
        f = np.fft.rfft(r, m)
        out[: lags + 1] = np.fft.irfft(f * np.conj(f), m)[: lags + 1]
    return out


def acf_tau(pieces, dt: float, max_lag_frac: float = 0.25, pair_weighted: bool = False) -> float:
    """
    Integrated autocorrelation time pooled over several pieces of a series.

    Each piece is a contiguous run of already-detrended (or de-meaned)
    values, such as fit residuals in one interval. Lag products are summed
    within pieces only, never across the boundaries between them, so gaps
    and interval edges add no spurious correlation. The estimator is the
    one of ``tau_int``: ``rho(k)`` is the mean lag-``k`` product over the
    mean square, summed up to the first non-positive ``rho`` or
    ``max_lag_frac`` of the longest piece, and ``tau = dt (1 + 2 sum)``,
    floored at ``dt``. Returns NaN when there is no data.

    With ``pair_weighted=True`` it returns the time constant *for the
    uncertainty of the mean of all pooled samples*. Pieces are taken as
    independent, so

        Var(mean) = s^2 / N * (1 + 2 sum_k (C_k / N) rho(k)),

    where ``C_k`` is the number of lag-``k`` pairs inside pieces and ``N``
    the number of samples. Each lag counts only for the samples that have
    such a partner. Many short pieces then contribute little correlation,
    whereas the unweighted estimate would apply it to every sample. For a
    single piece the weight is ``(n - k)/n``, the exact finite-sample
    factor. ``N_eff = N dt / tau`` then gives the mean's uncertainty.
    """
    pieces = [np.asarray(p, dtype=float) for p in pieces if len(p) > 0]
    if not pieces:
        return float("nan")
    lens = np.array([len(p) for p in pieces])
    max_lag = max(int(lens.max() * max_lag_frac), 1)
    sums = np.zeros(max_lag + 1)
    counts = np.zeros(max_lag + 1)
    ks = np.arange(max_lag + 1)
    for p in pieces:  # each piece adds only the lags it has, not max_lag of zeros
        s = _lag_sums(p, max_lag)
        m = len(s)
        sums[:m] += s
        counts[:m] += len(p) - ks[:m]
    var = sums[0] / counts[0]
    if not var > 0:
        return dt
    s = 0.0
    for k in range(1, max_lag + 1):
        if counts[k] == 0:
            break
        rho = sums[k] / counts[k] / var
        if rho <= 0:
            break
        s += rho * (counts[k] / counts[0] if pair_weighted else 1.0)
    return dt * max(1.0, 1.0 + 2.0 * s)


def mean_with_uncertainty(x: np.ndarray, dt: float, u_type_b: Optional[float] = None) -> dict:
    """
    Reported mean over an accepted interval with its uncertainty.

    The Type A part uses the effective sample count ``N_eff = duration /
    tau_int``, so that autocorrelated noise does not understate it:
    ``u_mean = s / sqrt(N_eff)``, with ``s`` the standard deviation of the
    detrended series. If ``u_type_b`` (a Type B standard uncertainty, the
    same for every sample) is given, it is combined in quadrature as in
    Zhang (2006, eq. 29): ``u_combined = sqrt(u_mean^2 + u_type_b^2)``.

    Returns a dict with ``mean``, ``u_mean``, ``u_combined``, ``tau_int``,
    ``n_eff`` and ``low_effective_sample`` (``N_eff < 10``, where a normal
    coverage factor is not appropriate).
    """
    x = np.asarray(x, dtype=float)
    tau = tau_int(x, dt)
    n_eff = max(len(x) * dt / tau, 1.0)
    tt = np.arange(len(x), dtype=float)
    _, _, r = linear_fit(tt, x)
    u_a = float(r.std(ddof=1) / np.sqrt(n_eff)) if len(x) > 1 else np.nan
    u_b = 0.0 if u_type_b is None else float(u_type_b)
    return {
        "mean": float(x.mean()),
        "u_mean": u_a,
        "u_combined": float(np.hypot(u_a, u_b)),
        "tau_int": tau,
        "n_eff": n_eff,
        "low_effective_sample": n_eff < 10,
    }

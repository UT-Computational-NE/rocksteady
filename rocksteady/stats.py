"""
Statistics shared by the criteria and by post-acceptance reporting.

The HAC slope estimator and the integrated autocorrelation time follow the
STNCL experimental plan (Newey-West with Bartlett weights, first
non-positive lag bandwidth, capped at a quarter of the record).
"""

from typing import Tuple

import numpy as np


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


def tau_int(x: np.ndarray, dt: float, max_lag_frac: float = 0.25) -> float:
    """Integrated autocorrelation time (same units as ``dt``) of the detrended series."""
    n = len(x)
    tt = np.arange(n, dtype=float)
    _, _, r = linear_fit(tt, np.asarray(x, dtype=float))
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


def mean_with_uncertainty(x: np.ndarray, dt: float) -> dict:
    """
    Reported mean over an accepted interval with its Type A uncertainty.

    Uses the effective sample count N_eff = duration / tau_int, so that
    autocorrelated noise does not understate the uncertainty.
    """
    x = np.asarray(x, dtype=float)
    tau = tau_int(x, dt)
    n_eff = max(len(x) * dt / tau, 1.0)
    tt = np.arange(len(x), dtype=float)
    _, _, r = linear_fit(tt, x)
    return {
        "mean": float(x.mean()),
        "u_mean": float(r.std(ddof=1) / np.sqrt(n_eff)) if len(x) > 1 else np.nan,
        "tau_int": tau,
        "n_eff": n_eff,
        "low_effective_sample": n_eff < 10,
    }

"""
Criteria: tests applied to one signal over the trailing window.

Each returns a ``CriterionResult`` (statistic, uncertainty, limit, k). The
detector turns that into a margin ratio and a three-way test, so a new
criterion only has to say what it measures and how well.

- ``DeviationCriterion``: the signal stays within a tolerance of a fitted
  model (constant or linear) over the window. Uncertainty comes from a
  measurement-noise model.
- ``DriftCriterion``: slope equivalence, ``|b| + k u_b <= b_crit``, with an
  autocorrelation-robust ``u_b`` estimated from the window itself (pluggable
  through ``registry.slope_methods``).
- ``BoundCriterion``: the window mean lies within a band. Uncertainty from
  the block-to-block scatter, corrected for autocorrelation.
- ``AsymptoteCriterion``: a first-order approach has reached its asymptote:
  the remaining approach, worst case over every time constant up to
  ``tau_max`` that the data allow, is within a limit.
"""

from typing import Callable, Mapping, Optional, Union

import numpy as np

from . import _validate as _v
from . import registry
from .core import Criterion, CriterionResult
from .stats import DEFAULT_SLOPE_METHOD, linear_fit, t_quantile, tau_int

NoiseModel = Union[float, Callable[[float], float]]


def _sigma(noise: NoiseModel, level: float) -> float:
    return float(noise(level)) if callable(noise) else float(noise)


@registry.criteria.register("deviation")
class DeviationCriterion(Criterion):
    """
    Max deviation of the block means from a fitted model within a tolerance.

    Parameters
    ----------
    signal : str
    limit : float
        Tolerance on the true max deviation (signal units, or relative if
        ``relative`` is True).
    noise : float or callable(level) -> float
        Standard deviation of a block mean due to measurement noise (same
        units as the limit). A callable receives the window mean, so noise
        can depend on level (for example, per power decade).
    model : "constant" or "linear"
        "constant": deviation from the window mean. "linear": deviation
        from the OLS line, which admits a steady drift (for example, an
        actuator slowly compensating for a slow process change).
    relative : bool
        Divide deviations by ``max(|window mean|, scale_floor)``.
    scale_floor : float
        Floor for the relative scale, so near-zero signals are judged on an
        absolute basis.
    k : float
        Width of the uncertainty interval in standard deviations.
    name : str, optional

    Notes
    -----
    The three-way test compares ``statistic +/- k * noise`` with the limit.
    This is a calibrated heuristic, not a confidence interval on the true
    maximum deviation: the maximum absolute residual of pure noise is
    itself biased upward (about 1.5 to 2 noise standard deviations for 10
    blocks), so ``limit`` and ``noise`` should be calibrated together on
    data known to be steady.
    """

    def __init__(self, signal: str, limit: float, noise: NoiseModel, model: str = "constant",
                 relative: bool = False, scale_floor: float = 0.0, k: float = 3.0, name: str = None):
        what = f"DeviationCriterion {name or signal!r}"
        if model not in ("constant", "linear"):
            raise ValueError(f"{what}: model must be 'constant' or 'linear', got {model!r}")
        limit = _v.finite(limit, f"{what}: limit", positive=True)
        if not callable(noise):
            noise = _v.finite(noise, f"{what}: noise", nonnegative=True)
        scale_floor = _v.finite(scale_floor, f"{what}: scale_floor", nonnegative=True)
        k = _v.finite(k, f"{what}: k", nonnegative=True)
        self.signal, self.limit, self.noise, self.model = signal, limit, noise, model
        self.relative, self.scale_floor, self.k = relative, scale_floor, k
        self.name = name or f"{signal}_{model}"

    def evaluate(self, t_blocks: np.ndarray, y_blocks: np.ndarray) -> CriterionResult:
        mean = float(y_blocks.mean())
        if self.model == "constant":
            resid = y_blocks - mean
        else:
            resid = linear_fit(t_blocks, y_blocks)[2]
        dev = float(np.max(np.abs(resid)))
        sigma = _sigma(self.noise, mean)  # relative when ``relative`` is True
        if self.relative:
            dev = dev / max(abs(mean), self.scale_floor)
        return CriterionResult(self.name, dev, sigma, self.limit, self.k)


@registry.criteria.register("drift")
class DriftCriterion(Criterion):
    """
    Slope equivalence test: ``|b| + k u_b <= limit``.

    The slope ``b`` and its uncertainty ``u_b`` come from the window's own
    block means, so no external noise model is needed.

    Parameters
    ----------
    signal : str
    limit_per_s : float
        Drift limit in signal units per second.
    k : float, optional
        Coverage factor. Default: the one supplied by the method (for
        ``"ar1"``, a Student-t quantile on the window's effective degrees of
        freedom; for ``"nw"``, 2).
    method : str or callable
        A name in ``registry.slope_methods`` or a callable
        ``(t, y, **method_options) -> (b, u_b, k)``. See ``rocksteady.stats``
        for the built-in methods and their calibration. ``"nw"`` reproduces
        rocksteady 0.1.
    method_options : mapping, optional
        Keyword arguments for the method (for example
        ``{"max_lag_frac": 0.25}`` for ``"nw"``).
    name : str, optional

    Needs at least 4 blocks.
    """

    min_blocks = 4

    def __init__(self, signal: str, limit_per_s: float, *, k: Optional[float] = None,
                 method: Union[str, Callable] = DEFAULT_SLOPE_METHOD,
                 method_options: Optional[Mapping] = None, name: Optional[str] = None):
        what = f"DriftCriterion {name or signal!r}"
        limit_per_s = _v.finite(limit_per_s, f"{what}: limit_per_s", positive=True)
        k = _v.optional_finite(k, f"{what}: k", nonnegative=True)
        self.signal, self.limit, self.k = signal, limit_per_s, k
        self.method = method
        self._slope = registry.slope_methods.get(method) if isinstance(method, str) else method
        if not callable(self._slope):
            raise TypeError(f"method must be a registered name or a callable, got {method!r}")
        self.method_options = dict(method_options or {})
        method_name = method if isinstance(method, str) else getattr(method, "__name__", "method")
        registry.check_options(self._slope, self.method_options, f"{what}: slope method {method_name!r}")
        self.name = name or f"{signal}_drift"

    def evaluate(self, t_blocks: np.ndarray, y_blocks: np.ndarray) -> CriterionResult:
        b, u_b, k = self._slope(t_blocks, y_blocks, **self.method_options)
        return CriterionResult(self.name, abs(b), u_b, self.limit, k if self.k is None else self.k)


@registry.criteria.register("bound")
class BoundCriterion(Criterion):
    """
    The window mean lies within ``[low, high]``.

    The statistic is the distance of the mean from the band's centre, and
    the limit is the half-width. The uncertainty is the standard error of
    the mean from block scatter, with the block count replaced by an
    effective count ``n / tau_int`` so that autocorrelated blocks do not
    understate it. With fewer than about 10 blocks the autocorrelation
    estimate, and so the uncertainty, is unreliable.
    """

    def __init__(self, signal: str, low: float, high: float, k: float = 2.0, name: str = None):
        what = f"BoundCriterion {name or signal!r}"
        low, high = _v.finite(low, f"{what}: low"), _v.finite(high, f"{what}: high")
        _v.ordered(low, high, f"{what}: low", "high", strict=True)
        k = _v.finite(k, f"{what}: k", nonnegative=True)
        self.signal, self.low, self.high, self.k = signal, low, high, k
        self.name = name or f"{signal}_bound"

    def evaluate(self, t_blocks: np.ndarray, y_blocks: np.ndarray) -> CriterionResult:
        centre, half = (self.low + self.high) / 2.0, (self.high - self.low) / 2.0
        mean = float(y_blocks.mean())
        n = len(y_blocks)
        if n > 1:
            n_eff = n / tau_int(y_blocks, 1.0)  # tau_int in blocks
            se = float(y_blocks.std(ddof=1) / np.sqrt(n_eff))
        else:
            se = 0.0
        return CriterionResult(self.name, abs(mean - centre), se, half, self.k)


@registry.criteria.register("asymptote")
class AsymptoteCriterion(Criterion):
    """
    A first-order approach has reached its asymptote: the remaining approach
    at the end of the window is within a limit.

    For a signal approaching a final value as
    ``y(t) = y_inf + A exp(-(t - t0) / tau)``, the remaining approach at the
    end of the window (the current sample) is

        d(tau) = A exp(-(t_end - t0) / tau) = -tau * y'(t_end).

    ``tau`` is rarely determined well by one window, and a test that trusts a
    fitted ``tau`` passes too often: noise that makes the curve look sharper
    makes ``tau`` look shorter and ``d`` look smaller. So the test takes the
    worst case over every ``tau`` the data allow (a profile-likelihood
    confidence set):

    1. For each ``tau`` on a log-spaced grid in ``[tau_min, tau_max]``, the
       model is linear in ``(y_inf, A)``. It is fitted by least squares,
       giving ``SSE(tau)`` and ``d(tau)``, with ``u(d(tau))`` from the AR(1)
       residual model, using the lag-1 autocorrelation of the best fit,
       bias-corrected for three parameters.
    2. The confidence set is ``SSE(tau) <= SSE_min (1 + t^2 / nu)``, with
       ``t`` the Student-t quantile at ``confidence`` and ``nu`` the effective
       degrees of freedom ``n (1 - phi) / (1 + phi) - 3``.
    3. The window passes when ``max over the set of |d(tau)| + k u(d(tau))``
       is within ``limit``, and fails when ``min over the set of
       |d(tau)| - k u(d(tau))`` exceeds it.

    ``tau_max`` is the caller's upper bound on the time constant, a property
    of the process rather than of the data. It is what makes every window
    decidable. When the window cannot resolve the approach (a flat window,
    or one much shorter than ``tau``), the set reaches ``tau_max``, where the
    model is a line with ``d ~ tau_max * |slope|``. That is the conservative
    bound for a monotone approach that slows down.

    Parameters
    ----------
    signal : str
    limit : float
        Largest acceptable remaining approach, in signal units (relative
        with ``relative=True``).
    tau_max : float
        Upper bound on the time constant of the approach, in seconds.
    tau_min : float, optional
        Lower end of the search. Default: half a block, or the window span
        divided by 10,000 if that is larger.
    k : float, optional
        Coverage factor. Default: the one-sided Student-t quantile at
        ``confidence`` on ``nu``.
    confidence : float
        For the default ``k`` and the confidence set.
    relative : bool
        Divide by ``max(|y_inf|, scale_floor)`` at the best-fitting ``tau``.
    scale_floor : float
    n_grid : int
        Points on the ``tau`` grid.
    name : str, optional

    Notes
    -----
    ``CriterionResult.details`` holds ``tau_s`` (the best-fitting ``tau``),
    ``tau_hi`` (the longest ``tau`` in the confidence set), ``y_inf`` (at the
    best fit) and ``remaining`` (the signed ``d`` at the best fit). The
    statistic and uncertainty encode the interval
    ``[min |d| - k u, max |d| + k u]`` over the set. Needs at least 10 blocks.
    """

    min_blocks = 10
    detail_keys = ("tau_s", "tau_hi", "y_inf", "remaining")

    def __init__(self, signal: str, limit: float, tau_max: float, *, tau_min: Optional[float] = None,
                 k: Optional[float] = None, confidence: float = 0.975, relative: bool = False,
                 scale_floor: float = 0.0, n_grid: int = 120, name: Optional[str] = None):
        what = f"AsymptoteCriterion {name or signal!r}"
        self.limit = _v.finite(limit, f"{what}: limit", positive=True)
        self.tau_max = _v.finite(tau_max, f"{what}: tau_max", positive=True)
        self.tau_min = _v.optional_finite(tau_min, f"{what}: tau_min", positive=True)
        _v.ordered(self.tau_min, self.tau_max, f"{what}: tau_min", "tau_max")
        self.k = _v.optional_finite(k, f"{what}: k", nonnegative=True)
        if not 0.5 < _v.finite(confidence, f"{what}: confidence") < 1:
            raise ValueError(f"{what}: confidence must be in (0.5, 1), got {confidence!r}")
        self.scale_floor = _v.finite(scale_floor, f"{what}: scale_floor", nonnegative=True)
        self.n_grid = _v.positive_int(n_grid, f"{what}: n_grid")
        self.signal, self.confidence, self.relative = signal, confidence, bool(relative)
        self.name = name or f"{signal}_asymptote"

    def evaluate(self, t_blocks: np.ndarray, y_blocks: np.ndarray) -> CriterionResult:
        from ._ar1 import ar1_apply

        t = np.asarray(t_blocks, dtype=float)
        y = np.asarray(y_blocks, dtype=float)
        n = len(y)
        step = float(t[1] - t[0]) if n > 1 else 1.0
        s = t - t[0]
        s_end = float(s[-1]) + 0.5 * step  # the end of the last block: the current sample
        lo = self.tau_min if self.tau_min is not None else max(0.5 * step, (s_end or 1.0) / 1e4)
        lo = min(lo, self.tau_max)
        grid = np.exp(np.linspace(np.log(lo), np.log(self.tau_max), self.n_grid))
        yc = y - y.mean()

        def fixed(taus):
            """For each tau: the 2-parameter fit y = y_inf + A g, g = exp(-s / tau), on a centred basis."""
            g = np.exp(-s[:, None] / taus[None, :])
            gc = g - g.mean(axis=0)
            sgg = np.einsum("ij,ij->j", gc, gc)
            ok = sgg > 1e-300
            a = np.where(ok, (gc.T @ yc) / np.where(ok, sgg, 1.0), 0.0)
            resid = yc[:, None] - gc * a[None, :]
            return gc, sgg, ok, a, y.mean() - a * g.mean(axis=0), resid, np.einsum("ij,ij->j", resid, resid)

        # The best tau, refined between the neighbours of the best grid point: residuals at a grid point
        # near a sharp optimum are systematic, and would read as strong autocorrelation.
        sse_grid = fixed(grid)[6]
        i = int(np.argmin(sse_grid))
        a_lo, a_hi = np.log(grid[max(i - 1, 0)]), np.log(grid[min(i + 1, len(grid) - 1)])
        gold = (np.sqrt(5.0) - 1.0) / 2.0
        c_, d_ = a_hi - gold * (a_hi - a_lo), a_lo + gold * (a_hi - a_lo)
        fc, fd = fixed(np.exp([c_]))[6][0], fixed(np.exp([d_]))[6][0]
        for _ in range(40):
            if fc < fd:
                a_hi, d_, fd = d_, c_, fc
                c_ = a_hi - gold * (a_hi - a_lo)
                fc = fixed(np.exp([c_]))[6][0]
            else:
                a_lo, c_, fc = c_, d_, fd
                d_ = a_lo + gold * (a_hi - a_lo)
                fd = fixed(np.exp([d_]))[6][0]
        t_best = float(np.exp(0.5 * (a_lo + a_hi)))
        if fixed(np.array([t_best]))[6][0] > sse_grid[i]:
            t_best = float(grid[i])
        taus = np.unique(np.append(grid, t_best))
        gc, sgg, ok, a_, y_inf, resid, sse = fixed(taus)
        best = int(np.flatnonzero(taus == t_best)[0])
        e = resid[:, best]
        ee = float(e @ e)
        r1 = float(e[1:] @ e[:-1]) / ee if ee > 0 else 0.0
        phi = min(max(r1 + (3.0 + 5.0 * r1) / n, 0.0), 0.98)  # bias correction for three fitted parameters
        nu = max(int(n * (1.0 - phi) / (1.0 + phi)) - 3, 1)
        tq = t_quantile(self.confidence, nu)
        k = tq if self.k is None else self.k
        in_set = ok & (sse <= sse[best] * (1.0 + tq * tq / nu) + 1e-300)
        # One grid point beyond each end: the set's true edges lie between grid points.
        in_set = ok & (in_set | np.r_[in_set[1:], False] | np.r_[False, in_set[:-1]])
        idx = np.flatnonzero(in_set)
        # u(A) at each tau in the set: Var(A) = sigma^2 gc'R gc / sgg^2, sigma^2 = SSE / tr(M R).
        rg = ar1_apply(gc[:, idx], phi)
        grg = np.einsum("ij,ij->j", gc[:, idx], rg)
        one_r_one = float(np.sum(ar1_apply(np.ones(n), phi))) / n
        tr_mr = n - one_r_one - grg / sgg[idx]
        sigma2 = sse[idx] / np.maximum(tr_mr, 1e-12)
        u_a = np.sqrt(np.maximum(sigma2 * grg, 0.0)) / sgg[idx]
        decay = np.exp(-s_end / taus[idx])
        d = a_[idx] * decay
        u_d = u_a * decay
        level = float(y_inf[best])
        scale = max(abs(level), self.scale_floor) if self.relative else 1.0
        upper = float(np.max(np.abs(d) + k * u_d)) / scale
        lower = max(float(np.min(np.abs(d) - k * u_d)), 0.0) / scale
        k_rep = k if k > 0 else 1.0
        details = {"tau_s": float(taus[best]), "tau_hi": float(taus[idx].max()), "y_inf": level,
                   "remaining": float(a_[best] * np.exp(-s_end / taus[best]))}
        return CriterionResult(self.name, 0.5 * (upper + lower), 0.5 * (upper - lower) / k_rep, self.limit,
                               float(k_rep), details)

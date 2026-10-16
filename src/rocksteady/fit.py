"""
Fitting a model to an interval.

A fit model describes the deterministic part of a signal over one interval
(a level, a ramp, an approach to a new value). Its residuals are the noise:
the fit *is* the detrend, so the choice of model decides what counts as
noise. Models are registered in ``registry.fit_models`` and called as
``model(t, y, **options) -> FitResult``.

Built in:

- ``"poly"``: polynomial of order ``order`` (0 = constant, 1 = line, ...).
  Parameters are the derivatives at the interval centre in physical units:
  ``level``, ``slope`` (per second), ``curvature`` (second derivative, per
  second squared), then ``d3``, ``d4``, ...
- ``"first_order"``: ``y_inf + (y0 - y_inf) exp(-(t - t0) / tau)``, an
  exponential rise or decay to an asymptote (``y0=0`` gives
  ``A (1 - exp(-lambda t))``).
- ``"exp"``: ``c + A exp((t - t0) / T)``, growth (T > 0) or decay (T < 0);
  ``T`` is a reactor period.

The exponential models are fitted by variable projection: for a fixed time
constant the model is linear in the other parameters, so a one-dimensional
search over the log time constant with a linear least-squares solve inside
finds the optimum (numpy only). A time constant is only determined when
the interval resolves the curve; ``tau_unidentifiable`` flags when it does
not, and the fit is still returned.

Model specs
-----------
For one signal, a model spec is an ``int`` (polynomial order), a model
name, or ``{"model": name, **options}``. It may be keyed by state, for
example ``{"steady": 0, "transient": 3}``. A missing ``"indeterminate"``
entry falls back to ``"transient"``.

Parameter uncertainty
---------------------
``cov`` is the parameter covariance with residuals modelled as AR(1):

    Cov = (J'J)^-1 J' (s^2 R) J (J'J)^-1,    s^2 = e'e / tr(M R)

where J is the Jacobian (the basis, for a polynomial), R the AR(1)
correlation matrix with the residuals' bias-corrected lag-1
autocorrelation, and M the residual-maker matrix. This generalizes
``stats.slope_ar1``. ``uncertainty="iid"`` uses R = I (the classical
result). Consecutive valid samples are treated as one sample apart.
The covariance is invariant to parameter units, and parameters the data
cannot determine get infinite variance (flag ``rank_deficient``).
"""

from dataclasses import dataclass, field
from math import factorial
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np

from . import _validate as _v
from ._ar1 import ar1_apply  # re-exported: rocksteady.fit.ar1_apply
from . import registry
from .states import Segment, State, check_segments

ModelSpec = Union[int, str, Mapping[str, Any]]

_STATE_KEYS = {s.value for s in State}


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class FitResult:
    """
    A model fitted over one interval.

    ``fitted`` and ``residuals`` (``y - fitted``) have the length of the
    input; ``residuals`` is NaN where ``y`` was not finite. ``params`` are
    in physical units and ``cov`` is their covariance, in the same order.
    """

    model: str
    params: Dict[str, float]
    cov: Optional[np.ndarray]
    fitted: np.ndarray
    residuals: np.ndarray
    flags: Set[str] = field(default_factory=set)
    options: Dict[str, Any] = field(default_factory=dict)
    phi: float = float("nan")  # lag-1 autocorrelation used for ``cov``
    n_params: int = 0  # free parameters fitted (degrees of freedom used)

    @property
    def u(self) -> Dict[str, float]:
        """Standard uncertainty of each parameter."""
        if self.cov is None:
            return {k: float("nan") for k in self.params}
        d = np.sqrt(np.clip(np.diag(self.cov), 0.0, None))
        return dict(zip(self.params, map(float, d)))

    @property
    def label(self) -> str:
        """Short description, e.g. ``"poly 1"``."""
        if self.model == "poly":
            return f"poly {self.options.get('order', 0)}"
        return self.model


def _empty(model: str, t: np.ndarray, names: Sequence[str], options: Dict[str, Any], flag: str) -> FitResult:
    nan = np.full(len(t), np.nan)
    return FitResult(model, {k: float("nan") for k in names}, None, nan, nan.copy(), {flag}, options)


# ---------------------------------------------------------------------------
# Autocorrelation-adjusted covariance
# ---------------------------------------------------------------------------


def residual_phi(e: np.ndarray, n_params: int, phi_max: float = 0.995) -> float:
    """
    Lag-1 autocorrelation of residuals, bias-corrected for ``n_params``
    fitted parameters and clipped to ``[0, phi_max]``.

    The correction ``(m + (m + 2) phi) / n`` reduces to Marriott and Pope's
    ``(1 + 3 phi) / n`` for a mean and to the ``(2 + 4 phi) / n`` used by
    ``stats.slope_ar1`` for a line.
    """
    n = len(e)
    ee = float(e @ e)
    if n < 3 or ee <= 0:
        return 0.0
    r1 = float(e[1:] @ e[:-1]) / ee
    return float(min(max(r1 + (n_params + (n_params + 2) * r1) / n, 0.0), phi_max))


def sandwich_cov(jac: np.ndarray, e: np.ndarray, method: str = "ar1", phi_max: float = 0.995,
                 rcond: float = 1e-10) -> Tuple[Optional[np.ndarray], float]:
    """
    Parameter covariance ``(J'J)^-1 J'(s^2 R)J (J'J)^-1`` for residuals ``e``.

    ``method`` is ``"ar1"`` or ``"iid"`` (R = I). Returns ``(cov, phi)``;
    ``cov`` is None when there are no residual degrees of freedom.

    The columns of J are scaled to unit norm and J is factorized by SVD, so
    the result does not depend on the parameters' units. A direction whose
    singular value is below ``rcond`` times the largest is not determined
    by the data. Every parameter with a component along such a direction
    gets an infinite variance (and NaN covariances), never a falsely small
    one.
    """
    n, m = jac.shape
    if n <= m:
        return None, float("nan")
    if method not in ("ar1", "iid"):
        raise ValueError(f"uncertainty must be 'ar1' or 'iid', got {method!r}")
    norms = np.linalg.norm(jac, axis=0)
    norms = np.where(norms > 0, norms, 1.0)  # an all-zero column shows up as a zero singular value
    u, sv, vt = np.linalg.svd(jac / norms, full_matrices=False)
    keep = sv > rcond * sv.max() if sv.max() > 0 else np.zeros(m, dtype=bool)
    rank = int(keep.sum())
    phi = residual_phi(e, max(rank, 1), phi_max) if method == "ar1" else 0.0
    uk = u[:, keep]
    g = uk.T @ ar1_apply(uk, phi)  # U'RU over the determined directions
    tr_mr = n - float(np.trace(g))
    if tr_mr <= 0:
        return None, phi
    s2 = float(e @ e) / tr_mr
    a_mat = vt[keep].T / sv[keep]  # V S^-1
    cov = s2 * (a_mat @ g @ a_mat.T)
    if rank < m:
        undetermined = np.abs(vt[~keep]).max(axis=0) > 1e-8
        cov[undetermined, :] = np.nan
        cov[:, undetermined] = np.nan
        cov[undetermined, undetermined] = np.inf
    return cov / np.outer(norms, norms), phi


def propagate_cov(cov: Optional[np.ndarray], jac: np.ndarray) -> Optional[np.ndarray]:
    """
    ``jac @ cov @ jac.T`` for a change of parameters, keeping infinite
    variances (undetermined parameters) infinite rather than turning
    ``0 * inf`` into NaN: an output that depends on an undetermined input
    is undetermined.
    """
    if cov is None:
        return None
    bad = ~np.isfinite(np.diag(cov))
    good = ~bad
    out = jac[:, good] @ cov[np.ix_(good, good)] @ jac[:, good].T
    if bad.any():
        dep = np.abs(jac[:, bad]).max(axis=1) > 0
        out[dep, :] = np.nan
        out[:, dep] = np.nan
        out[dep, dep] = np.inf
    return out


def _flag_undetermined(cov: Optional[np.ndarray], flags: Set[str]) -> None:
    if cov is not None and np.isinf(np.diag(cov)).any():
        flags.add("rank_deficient")


# ---------------------------------------------------------------------------
# Polynomial model
# ---------------------------------------------------------------------------


def _poly_names(order: int) -> List[str]:
    base = ["level", "slope", "curvature"]
    return [base[k] if k < 3 else f"d{k}" for k in range(order + 1)]


def _check_poly_options(order: int = 0, uncertainty: str = "ar1") -> None:
    if isinstance(order, bool) or int(order) != order or order < 0:
        raise ValueError(f"order must be a non-negative integer, got {order!r}")
    if uncertainty not in ("ar1", "iid"):
        raise ValueError(f"uncertainty must be 'ar1' or 'iid', got {uncertainty!r}")


def _check_time_bounds(lo, hi, lo_name: str, hi_name: str, uncertainty: str = "ar1") -> None:
    lo = _v.optional_finite(lo, lo_name, positive=True)
    hi = _v.optional_finite(hi, hi_name, positive=True)
    _v.ordered(lo, hi, lo_name, hi_name)
    if uncertainty not in ("ar1", "iid"):
        raise ValueError(f"uncertainty must be 'ar1' or 'iid', got {uncertainty!r}")


def _check_first_order_options(y0=None, tau_min=None, tau_max=None, uncertainty: str = "ar1") -> None:
    _v.optional_finite(y0, "y0")
    _check_time_bounds(tau_min, tau_max, "tau_min", "tau_max", uncertainty)


def _check_exp_options(T_min=None, T_max=None, uncertainty: str = "ar1") -> None:
    _check_time_bounds(T_min, T_max, "T_min", "T_max", uncertainty)


@registry.fit_models.register("poly")
def fit_poly(t: np.ndarray, y: np.ndarray, order: int = 0, uncertainty: str = "ar1") -> FitResult:
    """
    Least-squares polynomial of ``order`` over the interval.

    Time is scaled to [-1, 1] over the valid samples, so high orders stay
    well conditioned over long intervals. Non-finite samples are ignored.
    The order is reduced (flag ``order_reduced``) when there are fewer than
    ``order + 2`` valid samples.
    """
    try:
        _check_poly_options(order, uncertainty)
    except ValueError as exc:
        raise ValueError(f"poly: {exc}") from None
    order = int(order)
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    options = {"order": order, "uncertainty": uncertainty}
    ok = np.isfinite(t) & np.isfinite(y)
    nv = int(ok.sum())
    if nv == 0:
        return _empty("poly", t, _poly_names(order), options, "no_data")
    flags: Set[str] = set()
    p = min(order, max(nv - 2, 0))
    if p < order:
        flags.add("order_reduced")
    lo, hi = float(t[ok].min()), float(t[ok].max())
    centre, half = 0.5 * (lo + hi), 0.5 * (hi - lo) or 1.0
    x = (t - centre) / half
    basis = x[:, None] ** np.arange(p + 1)
    coef, *_ = np.linalg.lstsq(basis[ok], y[ok], rcond=None)
    fitted = basis @ coef
    resid = np.where(ok, y - fitted, np.nan)
    # Physical parameters: k-th derivative at the centre = k! coef_k / half^k.
    scale = np.array([factorial(k) / half**k for k in range(p + 1)])
    names = _poly_names(p)
    params = {k: float(v) for k, v in zip(names, scale * coef)}
    cov_c, phi = sandwich_cov(basis[ok], resid[ok], uncertainty)
    cov = propagate_cov(cov_c, np.diag(scale))
    _flag_undetermined(cov, flags)
    if p < order:  # report the requested parameters, missing ones as NaN
        for k in _poly_names(order)[p + 1:]:
            params[k] = float("nan")
        if cov is not None:
            full = np.full((order + 1, order + 1), np.nan)
            full[: p + 1, : p + 1] = cov
            cov = full
    options["order_used"] = p
    return FitResult("poly", params, cov, fitted, resid, flags, options, phi, p + 1)


# ---------------------------------------------------------------------------
# Exponential models (variable projection)
# ---------------------------------------------------------------------------

_GOLD = (np.sqrt(5.0) - 1.0) / 2.0


def _varpro_search(sse: Callable[[float], float], lo: float, hi: float,
                   n_grid: int = 80, iters: int = 60) -> Tuple[float, bool]:
    """
    Minimize ``sse(u)`` over ``u`` in ``[lo, hi]``: a log-spaced grid, then a
    golden-section refinement around the best grid point. Returns the best
    ``u`` and whether it sits on a bound of the range.
    """
    grid = np.linspace(lo, hi, n_grid)
    vals = np.array([sse(u) for u in grid])
    i = int(np.nanargmin(vals))
    a, b = grid[max(i - 1, 0)], grid[min(i + 1, n_grid - 1)]
    c, d = b - _GOLD * (b - a), a + _GOLD * (b - a)
    fc, fd = sse(c), sse(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - _GOLD * (b - a)
            fc = sse(c)
        else:
            a, c, fc = c, d, fd
            d = a + _GOLD * (b - a)
            fd = sse(d)
    u = 0.5 * (a + b)
    if sse(u) > vals[i]:
        u = grid[i]
    step = grid[1] - grid[0]
    return float(u), bool(u - lo < 0.5 * step or hi - u < 0.5 * step)


def _linear_ls(basis: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    """Least squares; a candidate whose basis is not finite (or fails to solve) scores infinitely badly."""
    if not np.all(np.isfinite(basis)):
        return np.full(basis.shape[1], np.nan), float("inf")
    try:
        coef, *_ = np.linalg.lstsq(basis, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.full(basis.shape[1], np.nan), float("inf")
    r = y - basis @ coef
    return coef, float(r @ r)


def _time_scales(s: np.ndarray, tau_min: Optional[float],
                 tau_max: Optional[float]) -> Tuple[float, float, float, float]:
    """(span, median spacing, lower and upper search bounds) for a time constant."""
    span = float(s.max())
    dt = float(np.median(np.diff(s))) if len(s) > 1 else 1.0
    lo = tau_min if tau_min is not None else max(0.5 * dt, span / 1e4)
    hi = tau_max if tau_max is not None else 100.0 * span
    if tau_min is None and lo > hi:  # an explicit upper bound below the default lower one
        lo = hi
    return span, dt, lo, hi


def _undetermine(cov: Optional[np.ndarray], idx: Sequence[int], flags: Set[str]) -> None:
    """Mark parameters ``idx`` as undetermined: infinite variance, NaN covariances."""
    if cov is not None:
        for i in idx:
            cov[i, :] = np.nan
            cov[:, i] = np.nan
            cov[i, i] = np.inf
    flags.add("rank_deficient")


def _negligible_amplitude(amplitude: float, y: np.ndarray) -> bool:
    """
    True when an exponential's amplitude is at the round-off level of the
    data values. The time constant of such a curve is undetermined, and the
    column scaling that makes covariances unit-invariant would otherwise
    turn round-off into an apparently determined direction.
    """
    return not abs(amplitude) > 1e3 * np.finfo(float).eps * float(np.max(np.abs(y)))


def _identifiability(flags: Set[str], tau: float, u_tau: float, span: float, dt: float) -> None:
    """Flag a time constant that the interval cannot determine."""
    if not np.isfinite(u_tau) or u_tau > 0.5 * abs(tau) or span < 0.5 * abs(tau) or abs(tau) < 5.0 * dt:
        flags.add("tau_unidentifiable")


@registry.fit_models.register("first_order")
def fit_first_order(t: np.ndarray, y: np.ndarray, y0: Optional[float] = None, tau_min: Optional[float] = None,
                    tau_max: Optional[float] = None, uncertainty: str = "ar1") -> FitResult:
    """
    First-order approach ``y = y_inf + (y0 - y_inf) exp(-(t - t0) / tau)``.

    ``t0`` is the first valid sample time. It covers exponential rise
    (``y_inf > y0``) and decay (``y_inf < y0``). Passing ``y0`` fixes the
    start value; ``y0=0`` gives ``A (1 - exp(-lambda t))`` with
    ``A = y_inf`` and ``lambda = 1 / tau``.

    Parameters reported: ``y0``, ``y_inf``, ``tau`` (s) and ``lambda`` (1/s),
    with uncertainties (``u(y0) = 0`` when fixed). Flags: ``tau_at_bound``,
    ``tau_unidentifiable`` (span shorter than tau/2, tau under five sample
    spacings, or ``u(tau) > tau/2``), ``too_few_samples``.
    """
    try:
        _check_first_order_options(y0, tau_min, tau_max, uncertainty)
    except ValueError as exc:
        raise ValueError(f"first_order: {exc}") from None
    names = ["y0", "y_inf", "tau", "lambda"]
    options = {"y0": y0, "tau_min": tau_min, "tau_max": tau_max, "uncertainty": uncertainty}
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(t) & np.isfinite(y)
    n_free = 2 if y0 is not None else 3
    if ok.sum() < n_free + 2:
        return _empty("first_order", t, names, options, "too_few_samples")
    t0 = float(t[ok].min())
    s_ok, y_ok = t[ok] - t0, y[ok]
    span, dt, lo, hi = _time_scales(s_ok, tau_min, tau_max)

    def basis(s, tau):
        g = np.exp(-s / tau)
        return np.column_stack((1.0 - g, g)) if y0 is None else (1.0 - g)[:, None]

    def sse(u):
        tau = np.exp(u)
        b = basis(s_ok, tau)
        yy = y_ok if y0 is None else y_ok - y0 * np.exp(-s_ok / tau)
        return _linear_ls(b, yy)[1]

    u, at_bound = _varpro_search(sse, np.log(lo), np.log(hi))
    tau = float(np.exp(u))
    yy = y_ok if y0 is None else y_ok - y0 * np.exp(-s_ok / tau)
    coef, _ = _linear_ls(basis(s_ok, tau), yy)
    y_inf = float(coef[0])
    y_start = float(coef[1]) if y0 is None else float(y0)
    s_all = t - t0
    g_all = np.exp(-s_all / tau)
    fitted = y_inf + (y_start - y_inf) * g_all
    resid = np.where(ok, y - fitted, np.nan)
    # Jacobian in theta = (y_inf, [y0,] log tau).
    g = np.exp(-s_ok / tau)
    d_logtau = (y_start - y_inf) * g * s_ok / tau
    cols = [1.0 - g] + ([g] if y0 is None else []) + [d_logtau]
    cov_th, phi = sandwich_cov(np.column_stack(cols), resid[ok], uncertainty)
    # params (y0, y_inf, tau, lambda) = h(theta)
    if y0 is None:
        jac_h = np.array([[0, 1, 0], [1, 0, 0], [0, 0, tau], [0, 0, -1.0 / tau]])
    else:
        jac_h = np.array([[0, 0], [1, 0], [0, tau], [0, -1.0 / tau]])
    cov = propagate_cov(cov_th, jac_h)
    flags: Set[str] = {"tau_at_bound"} if at_bound else set()
    _flag_undetermined(cov, flags)
    if _negligible_amplitude(y_start - y_inf, y_ok):
        _undetermine(cov, [2, 3], flags)  # tau, lambda
    u_tau = float(np.sqrt(max(cov[2, 2], 0.0))) if cov is not None else float("nan")
    _identifiability(flags, tau, u_tau, span, dt)
    params = {"y0": y_start, "y_inf": y_inf, "tau": tau, "lambda": 1.0 / tau}
    return FitResult("first_order", params, cov, fitted, resid, flags, options, phi, n_free)


@registry.fit_models.register("exp")
def fit_exp(t: np.ndarray, y: np.ndarray, T_min: Optional[float] = None, T_max: Optional[float] = None,
            uncertainty: str = "ar1") -> FitResult:
    """
    Free exponential ``y = c + A exp((t - t0) / T)``.

    ``T > 0`` is growth and ``T < 0`` decay; for reactor power, ``T`` is the
    period. ``t0`` is the first valid sample time, so ``A`` is the
    exponential part at the start. ``|T|`` is searched between ``T_min``
    (default: the larger of half a sample spacing and span / 300) and
    ``T_max`` (default 100 x span), for both signs. The basis is evaluated
    relative to the end of the interval for growth, so it never overflows,
    and a candidate that still gives non-finite values is skipped.

    Parameters reported: ``c``, ``A``, ``T`` (s). Flags as for
    ``first_order``.
    """
    try:
        _check_exp_options(T_min, T_max, uncertainty)
    except ValueError as exc:
        raise ValueError(f"exp: {exc}") from None
    names = ["c", "A", "T"]
    options = {"T_min": T_min, "T_max": T_max, "uncertainty": uncertainty}
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(t) & np.isfinite(y)
    if ok.sum() < 5:
        return _empty("exp", t, names, options, "too_few_samples")
    t0 = float(t[ok].min())
    s_ok, y_ok = t[ok] - t0, y[ok]
    span, dt, lo, hi = _time_scales(s_ok, T_min, T_max)
    lo = max(lo, span / 300.0) if T_min is None else lo

    s_end = float(s_ok.max())

    def ref(T):
        """Reference time for the basis exp((s - ref)/T), chosen so that it never exceeds 1."""
        return s_end if T > 0 else 0.0

    def basis(T):
        h = np.exp((s_ok - ref(T)) / T)
        return np.column_stack((np.ones_like(h), h))

    def fit_sign(sign):
        def sse(u):
            return _linear_ls(basis(sign * np.exp(u)), y_ok)[1]
        u, at_bound = _varpro_search(sse, np.log(lo), np.log(hi))
        return sse(u), sign * float(np.exp(u)), at_bound

    best = min((fit_sign(+1.0), fit_sign(-1.0)), key=lambda r: r[0])
    _, T, at_bound = best
    r0 = ref(T)
    h = np.exp((s_ok - r0) / T)
    coef, _ = _linear_ls(np.column_stack((np.ones_like(h), h)), y_ok)
    c, B = float(coef[0]), float(coef[1])  # B: the exponential part at the reference time
    A = B * float(np.exp(-r0 / T))  # ... and at the interval start (may underflow for steep growth)
    fitted = c + B * np.exp((t - t0 - r0) / T)
    resid = np.where(ok, y - fitted, np.nan)
    # Jacobian in theta = (c, B, log|T|); d/dlog|T| of B exp((s - r0)/T) is -B h (s - r0) / T.
    jac = np.column_stack((np.ones_like(h), h, -B * h * (s_ok - r0) / T))
    cov_th, phi = sandwich_cov(jac, resid[ok], uncertainty)
    # (c, A, T) from theta: A = B exp(-r0/T), T = sign * exp(log|T|).
    jac_h = np.array([[1.0, 0.0, 0.0], [0.0, float(np.exp(-r0 / T)), A * r0 / T], [0.0, 0.0, T]])
    cov = propagate_cov(cov_th, jac_h)
    flags: Set[str] = {"tau_at_bound"} if at_bound else set()
    _flag_undetermined(cov, flags)
    if _negligible_amplitude(B, y_ok):
        _undetermine(cov, [2], flags)  # T
    u_T = float(np.sqrt(max(cov[2, 2], 0.0))) if cov is not None else float("nan")
    _identifiability(flags, T, u_T, span, dt)
    return FitResult("exp", {"c": c, "A": A, "T": T}, cov, fitted, resid, flags, options, phi, 3)


# ---------------------------------------------------------------------------
# Model specs and piecewise fits
# ---------------------------------------------------------------------------


def resolve_model(spec: ModelSpec, state: Union[State, str, None] = None) -> Tuple[str, Dict[str, Any]]:
    """
    ``(model name, options)`` for a model spec, picking the entry for
    ``state`` when the spec is keyed by state.
    """
    if isinstance(spec, Mapping) and "model" not in spec:
        keys = set(spec)
        if not keys or not keys <= _STATE_KEYS:
            raise ValueError(f"a state-keyed model spec may only have keys {sorted(_STATE_KEYS)}, got {sorted(keys)}")
        st = State(state).value if state is not None else None
        if st is None:
            raise ValueError("this model spec is keyed by state; a state is needed")
        if st not in spec and st == State.INDETERMINATE.value:
            st = State.TRANSIENT.value
        if st not in spec:
            raise ValueError(f"model spec has no entry for state {st!r}: {dict(spec)!r}")
        return resolve_model(spec[st])
    if isinstance(spec, bool):
        raise ValueError(f"invalid model spec {spec!r}")
    if isinstance(spec, (int, np.integer)):
        if spec < 0:
            raise ValueError(f"polynomial order must be >= 0, got {spec}")
        return "poly", {"order": int(spec)}
    if isinstance(spec, str):
        registry.fit_models.get(spec)
        return spec, {}
    if isinstance(spec, Mapping):
        opts = dict(spec)
        name = opts.pop("model")
        registry.fit_models.get(name)
        return name, opts
    raise ValueError(f"invalid model spec {spec!r}")


fit_poly.validate_options = _check_poly_options
fit_first_order.validate_options = _check_first_order_options
fit_exp.validate_options = _check_exp_options


def check_model_options(name: str, options: Mapping[str, Any]) -> None:
    """
    Check a fit model's options before any data: the option names must be
    parameters of the provider, and a provider with a ``validate_options``
    attribute checks their values. Raises ``ValueError``.
    """
    import inspect

    provider = registry.fit_models.get(name)
    try:
        inspect.signature(provider).bind(np.zeros(0), np.zeros(0), **options)
    except TypeError as exc:
        raise ValueError(f"{name}: invalid options {sorted(options)} ({exc})") from None
    check = getattr(provider, "validate_options", None)
    if check is not None:
        try:
            check(**options)
        except ValueError as exc:
            raise ValueError(f"{name}: {exc}") from None


def validate_models(models: Mapping[str, ModelSpec],
                    states: Sequence[Union[State, str]] = tuple(State)) -> None:
    """
    Check a ``{signal: spec}`` mapping before any data: every state resolves
    to a registered model, and its options are valid (``check_model_options``).
    Errors name the signal, the state, the model and the option.
    """
    for name, spec in models.items():
        for st in states:
            try:
                model, opts = resolve_model(spec, st)
                check_model_options(model, opts)
            except (ValueError, KeyError) as exc:
                raise ValueError(f"model for {name!r} in state {State(st).value!r}: {exc}") from None


def fit_model(spec: ModelSpec, t: np.ndarray, y: np.ndarray, state: Union[State, str, None] = None) -> FitResult:
    """Fit one interval with the model named by ``spec`` (see ``resolve_model``)."""
    name, opts = resolve_model(spec, state)
    return registry.fit_models.get(name)(np.asarray(t, dtype=float), np.asarray(y, dtype=float), **opts)


@dataclass
class PiecewiseFit:
    """One fit per segment, and the full-length fitted curve."""

    fitted: np.ndarray
    segments: List[Segment]
    results: List[FitResult]

    @property
    def residuals(self) -> np.ndarray:
        out = np.full(len(self.fitted), np.nan)
        for s, r in zip(self.segments, self.results):
            out[s.slice] = r.residuals
        return out


def piecewise_fit(t: np.ndarray, y: np.ndarray, segs: Sequence[Segment], model: ModelSpec) -> PiecewiseFit:
    """
    Fit each segment with the model for its state.

    ``t`` and ``y`` are full-record arrays; ``segs`` index into them (for
    example from ``rocksteady.segments``). ``model`` is one signal's spec.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(t) != len(y):
        raise ValueError("t and y must have the same length")
    segs = check_segments(segs, len(y))
    fitted = np.full(len(y), np.nan)
    results = []
    for s in segs:
        r = fit_model(model, t[s.slice], y[s.slice], s.state)
        fitted[s.slice] = r.fitted
        results.append(r)
    return PiecewiseFit(fitted, list(segs), results)


__all__ = ["FitResult", "PiecewiseFit", "fit_poly", "fit_first_order", "fit_exp", "fit_model", "piecewise_fit",
           "resolve_model", "validate_models", "check_model_options", "sandwich_cov", "propagate_cov", "ar1_apply", "residual_phi"]

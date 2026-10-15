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
"""

from dataclasses import dataclass, field
from math import factorial
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np

from . import registry
from .states import Segment, State

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


def ar1_apply(x: np.ndarray, phi: float) -> np.ndarray:
    """
    ``R @ x`` for the AR(1) correlation matrix ``R_ij = phi^|i-j|``, in O(n).

    ``x`` may be a vector or an (n, m) matrix; columns are treated
    independently. Uses ``R x = f + b - x`` with the forward and backward
    recursions ``f_i = x_i + phi f_{i-1}`` and ``b_i = x_i + phi b_{i+1}``.
    """
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


def sandwich_cov(jac: np.ndarray, e: np.ndarray, method: str = "ar1",
                 phi_max: float = 0.995) -> Tuple[Optional[np.ndarray], float]:
    """
    Parameter covariance ``(J'J)^-1 J'(s^2 R)J (J'J)^-1`` for residuals ``e``.

    ``method`` is ``"ar1"`` or ``"iid"`` (R = I). Returns ``(cov, phi)``;
    ``cov`` is None when there are no residual degrees of freedom.
    """
    n, m = jac.shape
    if n <= m:
        return None, float("nan")
    if method not in ("ar1", "iid"):
        raise ValueError(f"uncertainty must be 'ar1' or 'iid', got {method!r}")
    phi = residual_phi(e, m, phi_max) if method == "ar1" else 0.0
    jtj_inv = np.linalg.pinv(jac.T @ jac)
    rj = ar1_apply(jac, phi)
    jrj = jac.T @ rj
    tr_mr = n - float(np.trace(jtj_inv @ jrj))
    if tr_mr <= 0:
        return None, phi
    s2 = float(e @ e) / tr_mr
    return s2 * (jtj_inv @ jrj @ jtj_inv), phi


# ---------------------------------------------------------------------------
# Polynomial model
# ---------------------------------------------------------------------------


def _poly_names(order: int) -> List[str]:
    base = ["level", "slope", "curvature"]
    return [base[k] if k < 3 else f"d{k}" for k in range(order + 1)]


@registry.fit_models.register("poly")
def fit_poly(t: np.ndarray, y: np.ndarray, order: int = 0, uncertainty: str = "ar1") -> FitResult:
    """
    Least-squares polynomial of ``order`` over the interval.

    Time is scaled to [-1, 1] over the valid samples, so high orders stay
    well conditioned over long intervals. Non-finite samples are ignored.
    The order is reduced (flag ``order_reduced``) when there are fewer than
    ``order + 2`` valid samples.
    """
    if int(order) != order or order < 0:
        raise ValueError(f"poly order must be a non-negative integer, got {order!r}")
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
    cov = None if cov_c is None else cov_c * np.outer(scale, scale)
    if p < order:  # report the requested parameters, missing ones as NaN
        for k in _poly_names(order)[p + 1:]:
            params[k] = float("nan")
        if cov is not None:
            full = np.full((order + 1, order + 1), np.nan)
            full[: p + 1, : p + 1] = cov
            cov = full
    options["order_used"] = p
    return FitResult("poly", params, cov, fitted, resid, flags, options, phi)


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


def validate_models(models: Mapping[str, ModelSpec],
                    states: Sequence[Union[State, str]] = tuple(State)) -> None:
    """Check a ``{signal: spec}`` mapping resolves for every state; raise ``ValueError`` otherwise."""
    for name, spec in models.items():
        for st in states:
            try:
                resolve_model(spec, st)
            except (ValueError, KeyError) as exc:
                raise ValueError(f"model for {name!r}: {exc}") from None


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
    fitted = np.full(len(y), np.nan)
    results = []
    for s in segs:
        r = fit_model(model, t[s.slice], y[s.slice], s.state)
        fitted[s.slice] = r.fitted
        results.append(r)
    return PiecewiseFit(fitted, list(segs), results)


__all__ = ["FitResult", "PiecewiseFit", "fit_poly", "fit_model", "piecewise_fit", "resolve_model",
           "validate_models", "sandwich_cov", "ar1_apply", "residual_phi"]

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
"""

from typing import Callable, Mapping, Optional, Union

import numpy as np

from . import _validate as _v
from . import registry
from .core import Criterion, CriterionResult
from .stats import DEFAULT_SLOPE_METHOD, linear_fit, tau_int

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
        Divide deviations by max(|window mean|, ``scale_floor``).
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

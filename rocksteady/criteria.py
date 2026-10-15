"""
Criteria: tests applied to one signal over the trailing window.

Each returns a ``CriterionResult`` (statistic, uncertainty, limit, k). The
detector turns that into a margin ratio and a three-way test, so a new
criterion only has to say what it measures and how well.

- ``DeviationCriterion``: the signal stays within a tolerance of a fitted
  model (constant or linear) over the window. Uncertainty comes from a
  measurement-noise model. This is the reactor power and rod test.
- ``DriftCriterion``: slope equivalence, ``|b| + k u_b <= b_crit``, with an
  autocorrelation-robust ``u_b`` estimated from the window itself. This is
  the flow-loop drift test.
- ``BoundCriterion``: the window mean lies within a band. Uncertainty from
  the block-to-block scatter, corrected for autocorrelation.
"""

from typing import Callable, Union

import numpy as np

from .core import Criterion, CriterionResult
from .stats import hac_slope, linear_fit, tau_int

NoiseModel = Union[float, Callable[[float], float]]


def _sigma(noise: NoiseModel, level: float) -> float:
    return float(noise(level)) if callable(noise) else float(noise)


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
        from the OLS line, which admits a steady drift (the rod exemption).
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
        if model not in ("constant", "linear"):
            raise ValueError("model must be 'constant' or 'linear'")
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


class DriftCriterion(Criterion):
    """
    Slope equivalence test: ``|b| + k u_b <= limit``.

    ``u_b`` is a Newey-West standard error from the window's own residuals,
    so no external noise model is needed. ``limit`` is in signal units per
    second. Needs at least 4 blocks.
    """

    min_blocks = 4

    def __init__(self, signal: str, limit_per_s: float, k: float = 2.0, max_lag_frac: float = 0.25,
                 name: str = None):
        self.signal, self.limit, self.k, self.max_lag_frac = signal, limit_per_s, k, max_lag_frac
        self.name = name or f"{signal}_drift"

    def evaluate(self, t_blocks: np.ndarray, y_blocks: np.ndarray) -> CriterionResult:
        b, u_b = hac_slope(t_blocks, y_blocks, self.max_lag_frac)
        return CriterionResult(self.name, abs(b), u_b, self.limit, self.k)


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

"""
Statistics over intervals and states, with uncertainties corrected for
autocorrelation.

Neighbouring samples are correlated, so ``std / sqrt(n)`` understates the
uncertainty of a mean or a bias. These statistics use an effective sample
count ``N_eff = duration / tau_int`` instead, where ``tau_int`` is the
integrated autocorrelation time estimated from the data
(``stats.acf_tau``). The spread itself (``std``) is not corrected; the
uncertainty of a mean is.

Three kinds of statistics:

- ``signal_stats``: one live signal over one interval, about the
  interval's fit. The fit is the detrend: ``std``, ``tau_int`` and
  ``N_eff`` describe the residuals, and ``u_mean = std / sqrt(N_eff)``.
- ``error_stats``: one series compared with another, ``e = a - b`` (a fit
  vs the data, or a model vs a measurement). The error is de-meaned but
  not detrended, because slow wander in an offset is real uncertainty in
  the bias: ``u_bias = std / sqrt(N_eff)``.
- ``stats_by_state`` / ``stats_by_segment``: ``error_stats`` pooled over
  the segments of each state, or per segment.

Autocorrelation is estimated within contiguous pieces only: never across
segment boundaries or across non-finite samples.
"""

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .fit import FitResult
from .states import Segment, State
from .stats import acf_tau

NRMSE_NORMS = ("mean", "range", "std", "none")
ALL = "all"
STATES = (ALL,) + tuple(s.value for s in State)


def finite_pieces(mask: np.ndarray, bounds: Optional[Sequence[Tuple[int, int]]] = None) -> List[Tuple[int, int]]:
    """
    Contiguous runs ``[start, stop)`` where ``mask`` is True, within each of
    ``bounds`` (default: the whole array).
    """
    mask = np.asarray(mask, dtype=bool)
    out: List[Tuple[int, int]] = []
    for lo, hi in bounds if bounds is not None else [(0, len(mask))]:
        m = mask[lo:hi]
        if not len(m):
            continue
        d = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
        out += [(lo + int(a), lo + int(b)) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]
    return out


# ---------------------------------------------------------------------------
# One live signal over an interval
# ---------------------------------------------------------------------------


@dataclass
class SignalStats:
    """Statistics of one signal over one interval, about the interval's fit."""

    n: int
    duration_s: float
    mean: float
    std: float
    tau_int_s: float
    n_eff: float
    u_mean: float
    low_effective_sample: bool
    model: str = ""
    params: Dict[str, float] = field(default_factory=dict)
    u_params: Dict[str, float] = field(default_factory=dict)
    flags: Set[str] = field(default_factory=set)

    def as_dict(self) -> dict:
        return asdict(self)


def signal_stats(y: np.ndarray, fit: FitResult, *, dt: float, max_lag_frac: float = 0.25) -> SignalStats:
    """
    Statistics of ``y`` over one interval, given the interval's ``fit``.

    ``std`` is the residual standard deviation with the fit's degrees of
    freedom removed; ``tau_int`` and ``N_eff`` come from the residuals; and
    ``u_mean = std / sqrt(N_eff)`` is the uncertainty of the interval mean.
    """
    y = np.asarray(y, dtype=float)
    e = np.asarray(fit.residuals, dtype=float)
    ok = np.isfinite(e) & np.isfinite(y)
    n = int(ok.sum())
    extra = dict(model=fit.label, params=dict(fit.params), u_params=fit.u, flags=set(fit.flags))
    if n == 0:
        nan = float("nan")
        return SignalStats(0, 0.0, nan, nan, nan, nan, nan, True, **extra)
    dof = max(n - max(fit.n_params, 1), 1)
    std = float(np.sqrt(np.sum(e[ok] ** 2) / dof))
    tau = acf_tau([e[a:b] for a, b in finite_pieces(ok)], dt, max_lag_frac)
    n_eff = max(n * dt / tau, 1.0)
    return SignalStats(n, n * dt, float(y[ok].mean()), std, tau, n_eff, std / np.sqrt(n_eff),
                       n_eff < 10, **extra)


# ---------------------------------------------------------------------------
# Comparing two series
# ---------------------------------------------------------------------------


@dataclass
class ErrorStats:
    """Comparison of ``a`` with ``b`` (``e = a - b``) over a set of samples."""

    n: int
    duration_s: float
    bias: float
    std: float
    mae: float
    rmse: float
    max_abs: float
    nrmse: float
    r: float
    tau_int_s: float
    n_eff: float
    u_bias: float
    low_effective_sample: bool

    def as_dict(self) -> dict:
        return asdict(self)


def _nan_error_stats() -> ErrorStats:
    nan = float("nan")
    return ErrorStats(0, 0.0, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, True)


def error_stats(a: np.ndarray, b: np.ndarray, *, dt: float, pieces: Optional[Sequence[Tuple[int, int]]] = None,
                nrmse: str = "mean", max_lag_frac: float = 0.25) -> ErrorStats:
    """
    Error statistics of ``a`` against the reference ``b``, with ``e = a - b``.

    Parameters
    ----------
    a, b : arrays of the same length
        Samples where either is not finite are ignored.
    dt : float
        Sample period in seconds.
    pieces : sequence of (start, stop), optional
        Index ranges that are each contiguous in time (for example the
        segments of one state). Only samples inside them are used, and
        autocorrelation is never estimated across their boundaries.
        Default: the whole array.
    nrmse : "mean" (default), "range", "std" or "none"
        Normalisation of NRMSE: ``|mean(b)|``, ``max(b) - min(b)``,
        ``std(b)``, or not computed (NaN).

    ``r`` is Pearson's correlation of ``a`` and ``b`` (NaN if either is
    constant). ``u_bias = std / sqrt(N_eff)`` with ``N_eff = n dt /
    tau_int`` from the de-meaned error.
    """
    if nrmse not in NRMSE_NORMS:
        raise ValueError(f"nrmse must be one of {NRMSE_NORMS}, got {nrmse!r}")
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")
    ok = np.isfinite(a) & np.isfinite(b)
    runs = finite_pieces(ok, pieces)
    if not runs:
        return _nan_error_stats()
    idx = np.concatenate([np.arange(lo, hi) for lo, hi in runs])
    av, bv = a[idx], b[idx]
    e = av - bv
    n = len(e)
    bias = float(e.mean())
    std = float(e.std(ddof=1)) if n > 1 else float("nan")
    rmse = float(np.sqrt(np.mean(e * e)))
    norm = {"mean": abs(float(bv.mean())), "range": float(np.ptp(bv)),
            "std": float(bv.std(ddof=1)) if n > 1 else float("nan"), "none": float("nan")}[nrmse]
    sa, sb = float(av.std()), float(bv.std())
    r = float(np.mean((av - av.mean()) * (bv - bv.mean())) / (sa * sb)) if sa > 0 and sb > 0 else float("nan")
    tau = acf_tau([a[lo:hi] - b[lo:hi] - bias for lo, hi in runs], dt, max_lag_frac)
    n_eff = max(n * dt / tau, 1.0)
    return ErrorStats(
        n=n, duration_s=n * dt, bias=bias, std=std, mae=float(np.mean(np.abs(e))), rmse=rmse,
        max_abs=float(np.max(np.abs(e))), nrmse=rmse / norm if norm > 0 else float("nan"), r=r,
        tau_int_s=tau, n_eff=n_eff, u_bias=std / np.sqrt(n_eff) if n > 1 else float("nan"),
        low_effective_sample=bool(n_eff < 10),
    )


def stats_by_state(a: np.ndarray, b: np.ndarray, segs: Sequence[Segment], *, dt: float,
                   states: Sequence[str] = STATES, nrmse: str = "mean", max_lag_frac: float = 0.25):
    """
    ``error_stats`` per state: one row per entry of ``states`` ("all",
    "steady", "transient", "indeterminate"), pooling that state's segments.
    Returns a DataFrame indexed by state.
    """
    import pandas as pd

    rows = {}
    for st in states:
        sel = [(s.start, s.stop) for s in segs if st == ALL or s.state == State(st)]
        es = error_stats(a, b, dt=dt, pieces=sel, nrmse=nrmse, max_lag_frac=max_lag_frac) if sel else _nan_error_stats()
        rows[str(st)] = es.as_dict()
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "state"
    return out


def stats_by_segment(a: np.ndarray, b: np.ndarray, segs: Sequence[Segment], *, dt: float,
                     nrmse: str = "mean", max_lag_frac: float = 0.25):
    """``error_stats`` for each segment. Returns a DataFrame with the segment's index, state, start and stop."""
    import pandas as pd

    rows = []
    for i, s in enumerate(segs):
        es = error_stats(a, b, dt=dt, pieces=[(s.start, s.stop)], nrmse=nrmse, max_lag_frac=max_lag_frac)
        rows.append({"segment": i, "state": s.state.value, "start": s.start, "stop": s.stop, **es.as_dict()})
    return pd.DataFrame(rows)


__all__ = ["SignalStats", "ErrorStats", "signal_stats", "error_stats", "stats_by_state", "stats_by_segment",
           "finite_pieces", "STATES"]

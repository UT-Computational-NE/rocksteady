"""
Best-fit segments under limits: constrained optimal segmentation.

``optimal_segments`` splits a record into segments, gives each a state
(``steady``, ``transient`` or ``indeterminate``), and fits each with the model
for its state. It chooses the breakpoints and states that minimise

    sum over segments of (SSE of the segment's fit + penalty of its state)

subject to the caller's limits. That is the best piecewise approximation of
the record, in the RMS sense, that the limits allow.

This answers a different question from the detector. The detector certifies,
causally, that a signal *is* steady within a limit. This optimiser describes
a finished record as well as possible with a given budget of segments. It will
use the segments it is allowed whether or not the data would pass the
detector's criteria. Its segments are "best-fit segments", not certified
steady states; ``compare_segments`` measures how far they agree with the
detector's (``states.segments_from``).

Limits
------
- ``per`` and ``max_counts``: at most ``max_counts[state]`` segments of a
  state per period (a day, a week, ...). A segment counts once against
  every period it touches, so one segment from 16:00 to 08:00 the next day
  counts against both days. Periods are half-open, and a segment touches a
  period when one of its samples lies in it. For a ``DatetimeIndex``,
  periods follow its wall clock (local days for a local time zone). When
  the clock falls back an hour, the repeated wall time stays in the period
  already in progress, so a period shorter than an hour is not counted
  twice; days are unaffected.
- ``transitions``: which state may follow which. Default: any. This
  constrains the sequence of labels only. Segments are fitted independently,
  so consecutive fits need not meet: a level change can sit at any boundary.
  Alternation forbids steady -> steady, but the optimiser may then place a
  short transient on a plateau next to a step rather than across it.
- ``min_duration_s`` / ``max_duration_s``: per state. The duration of a
  segment is from its first to its last sample, plus one sample spacing.
- ``penalty``: added for every segment, per state. With 0, more segments
  never fit worse, so every allowed slot is used. The default, ``"bic"``,
  charges ``sigma^2 (p + 1) ln n`` for a model with ``p`` parameters,
  ``sigma`` estimated from the first differences of the data (robustly), so
  "at most" stays "at most".

Methods
-------
All three run one dynamic program over candidate breakpoints, which tracks
the last segment's state and the counts used in the current period.

- ``"subset"``: candidates are the boundaries given in ``candidates`` (for
  example the detector's segments). Optimal among those.
- ``"refine"``: ``subset``, then each kept boundary is moved within
  ``refine_s`` to lower the total cost, keeping states and limits. A local
  optimum near the candidates; never worse than ``subset``.
- ``"exact"``: candidates are every sample, or the first sample of every
  ``grid_s`` interval, plus ``candidates``. The global optimum on that grid.

Optimal breakpoints for fewer segments are not a subset of those for more
(fit ``y = t`` with constant segments: the best two-segment split is at 0.5,
the best three-segment split at 1/3 and 2/3), so ``subset`` is an
approximation unless the candidates contain the optimum.

Cost
----
Constant and linear (``poly`` order 0 or 1) segment costs come from range
sums, so they are cheap even for ``"exact"``. The sums carry an error bound
(``_RangeSum``); where rounding could hide a segment's residual (a quiet
segment next to a much larger level), that segment is fitted directly
instead. Any other model is fitted with ``fit.fit_model``, exactly as
``report.fit_report`` will fit it, and dominates the run time;
``max_duration_s`` is the main lever on it.

The search drops a candidate segment once its own cost exceeds the cost of a
known feasible solution (``prune=True``). That is exact when a longer segment
never fits better than a shorter one it contains, which holds for least
squares once the model is fitted in full. So pruning waits while a
polynomial is still fitted with a reduced order (``order_reduced``, too few
samples), and segments too short for a model never stop the search. The
exponential models' default time-constant bounds scale with the segment's
span. Give explicit bounds (``T_min``/``T_max``, ``tau_min``/``tau_max``)
for a strict guarantee, or pass ``prune=False``.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from . import fit as _fit
from .states import Segment, State, check_segments

METHODS = ("subset", "refine", "exact")


# ---------------------------------------------------------------------------
# Times and periods
# ---------------------------------------------------------------------------


def _times(t) -> Tuple[np.ndarray, Any]:
    """Seconds from the first sample, and the DatetimeIndex (or None)."""
    if hasattr(t, "tz_localize") or (hasattr(t, "dtype") and np.issubdtype(getattr(t, "dtype", float), np.datetime64)):
        import pandas as pd

        idx = pd.DatetimeIndex(t)
        if len(idx) == 0:
            return np.zeros(0), idx
        return (idx - idx[0]).total_seconds().to_numpy(dtype=float), idx
    return np.asarray(t, dtype=float), None


def _period_ids(ts: np.ndarray, idx, per, period_origin) -> Optional[np.ndarray]:
    """Period number of every sample, or None for a single period (``per=None``)."""
    if per is None:
        return None
    if idx is not None:
        import pandas as pd

        wall = idx.tz_localize(None) if idx.tz is not None else idx  # local wall-clock time
        origin = pd.Timestamp("1970-01-01") if period_origin is None else pd.Timestamp(period_origin)
        if origin.tzinfo is not None:
            origin = origin.tz_localize(None)
        step = pd.Timedelta(per)
        if step <= pd.Timedelta(0):
            raise ValueError(f"per must be positive, got {per!r}")
        ids = ((wall - origin) // step).to_numpy().astype(np.int64)
        # When the clock falls back (end of daylight saving), wall time repeats an hour, and a period shorter
        # than that would recur. Period numbers never go back: the repeated time stays in the period in
        # progress, so each period has one count budget, and the optimiser's counters stay valid.
        return np.maximum.accumulate(ids) if len(ids) else ids
    if isinstance(per, str):
        raise ValueError("per as a string (e.g. '1D') needs times as a DatetimeIndex; give seconds for numeric times")
    step = float(per)
    if not step > 0 or not math.isfinite(step):
        raise ValueError(f"per must be a positive number of seconds, got {per!r}")
    origin = 0.0 if period_origin is None else float(period_origin)
    return np.floor((ts - origin) / step).astype(np.int64)


def count_by_period(segments: Sequence[Segment], t, per, *, period_origin=None,
                    states: Optional[Sequence[str]] = None):
    """
    Segments of each state touching each period: a DataFrame indexed by
    period start, one column per state (``states``, default: those present).
    A segment counts once against every period one of its samples lies in.
    """
    import pandas as pd

    ts, idx = _times(t)
    segs = check_segments(segments, len(ts))
    pid = _period_ids(ts, idx, per, period_origin)
    if pid is None:
        pid = np.zeros(len(ts), dtype=np.int64)
    rows: Dict[int, Dict[str, int]] = {}
    for s in segs:
        for p in range(int(pid[s.start]), int(pid[s.stop - 1]) + 1):
            if p not in rows:
                rows[p] = {}
            rows[p][s.state.value] = rows[p].get(s.state.value, 0) + 1
    # Periods inside a segment hold no sample of their own only when a gap spans them; they still count.
    states = list(states) if states is not None else sorted({s.state.value for s in segs})
    out = pd.DataFrame.from_dict(rows, orient="index", columns=states).fillna(0).astype(int).sort_index()
    if per is None:
        out.index = ["all"]
    elif idx is not None:
        origin = pd.Timestamp("1970-01-01") if period_origin is None else pd.Timestamp(period_origin).tz_localize(None)
        out.index = origin + out.index.to_numpy() * pd.Timedelta(per)
    else:
        out.index = (0.0 if period_origin is None else float(period_origin)) + out.index.to_numpy() * float(per)
    out.index.name = "period"
    return out


# ---------------------------------------------------------------------------
# Segment costs
# ---------------------------------------------------------------------------


_EPS = float(np.finfo(float).eps)
_REDUCED_FLAGS = frozenset({"order_reduced"})  # the fit used fewer parameters than the model has


class _RangeSum:
    """
    Sums of ``x`` over ``[p, q)`` from prefix sums, with an error bound.

    Prefix sums lose a short range's sum when the record elsewhere is much
    larger. Here the prefix is kept in blocks of ``B`` samples: a running sum
    within each block, and the block totals accumulated with compensated
    (Neumaier) summation. The error of a range sum is then bounded by the
    range's own magnitude and the magnitudes of the two blocks at its ends,
    not by the whole record before it. ``range`` returns the sum and that bound.
    """

    B = 64

    def __init__(self, x: np.ndarray):
        n = len(x)
        nb = n // self.B + 1  # so that index n is in range
        xp = np.zeros(nb * self.B)
        xp[:n] = x
        blocks = xp.reshape(nb, self.B)
        local = np.cumsum(blocks, axis=1)
        self.loc = np.concatenate((np.zeros((nb, 1)), local[:, :-1]), axis=1).ravel()  # within-block, exclusive
        self.mass = np.abs(blocks).sum(axis=1)
        self.hi = np.zeros(nb)
        self.lo = np.zeros(nb)
        acc, comp = 0.0, 0.0
        for b, v in enumerate(local[:, -1].tolist()):  # exclusive block prefix, compensated
            self.hi[b], self.lo[b] = acc, comp
            t = acc + v
            comp += (acc - t) + v if abs(acc) >= abs(v) else (v - t) + acc
            acc = t

    def range(self, p, q):
        bp, bq = p // self.B, q // self.B
        v = (self.hi[bq] - self.hi[bp]) + ((self.lo[bq] - self.lo[bp]) + (self.loc[q] - self.loc[p]))
        err = 4.0 * _EPS * (np.abs(v) + self.B * (self.mass[bp] + self.mass[bq]))
        return v, err


class _Costs:
    """
    SSE of each state's model over samples ``[p, q)``. Constant and linear
    polynomials use range sums (vectorised over many starts) with an error
    bound; where cancellation could hide the residual, the segment is fitted
    directly instead. Other models are fitted with ``fit.fit_model`` and
    cached.
    """

    def __init__(self, ts: np.ndarray, y: np.ndarray, models: Mapping[str, Any], states: Sequence[str]):
        self.ts, self.y, self.models, self.states = ts, y, models, list(states)
        ok = np.isfinite(y) & np.isfinite(ts)
        self.ok = ok
        span = (ts[-1] - ts[0]) if len(ts) > 1 else 1.0
        mid = 0.5 * (ts[0] + ts[-1]) if len(ts) else 0.0
        x = np.where(ok, (ts - mid) / (0.5 * span if span > 0 else 1.0), 0.0)
        yc = np.where(ok, y - (np.mean(y[ok]) if ok.any() else 0.0), 0.0)
        self.N = np.concatenate(([0.0], np.cumsum(ok.astype(float))))  # integers: exact
        self.Sx, self.Sy = _RangeSum(x), _RangeSum(yc)
        self.Sxx, self.Sxy, self.Syy = _RangeSum(x * x), _RangeSum(x * yc), _RangeSum(yc * yc)
        self.fast: Dict[str, Optional[int]] = {}
        for st in self.states:
            name, opts = _fit.resolve_model(models[st])
            order = opts.get("order", 0) if name == "poly" else None
            self.fast[st] = int(order) if name == "poly" and int(order) <= 1 else None
        self.cache: Dict[Tuple[str, int, int], float] = {}
        self.reduced: Dict[Tuple[str, int, int], bool] = {}
        self.n_fits = 0
        self.n_direct = 0

    def is_fast(self, st: str) -> bool:
        return self.fast[st] is not None

    def _direct(self, order: int, p: int, q: int) -> float:
        """Constant or linear least-squares SSE of ``[p, q)``, computed on the segment itself."""
        self.n_direct += 1
        m = self.ok[p:q]
        yy = self.y[p:q][m]
        if len(yy) == 0:
            return 0.0
        if order == 0 or len(yy) < 3:  # fit_poly's order reduction
            d = yy - yy.mean()
            return float(d @ d)
        tt = self.ts[p:q][m]
        xx = tt - tt.mean()
        if not np.any(xx):
            d = yy - yy.mean()
            return float(d @ d)
        a = np.column_stack((np.ones_like(xx), xx / np.max(np.abs(xx))))
        coef, *_ = np.linalg.lstsq(a, yy, rcond=None)
        r = yy - a @ coef
        return float(r @ r)

    def vector(self, st: str, ps: np.ndarray, q: int) -> np.ndarray:
        """Costs of ``[p, q)`` for an array of starts ``ps`` (fast models only)."""
        ps = np.asarray(ps, dtype=np.int64)
        n = self.N[q] - self.N[ps]
        nn = np.where(n > 0, n, 1.0)
        syy, e_yy = self.Syy.range(ps, q)
        sy, e_y = self.Sy.range(ps, q)
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            sse = np.where(n > 0, syy - sy * sy / nn, 0.0)
            err = e_yy + (2.0 * np.abs(sy) * e_y + e_y * e_y) / nn + 4.0 * _EPS * (np.abs(syy) + sy * sy / nn)
            if self.fast[st] == 1:
                sx, e_x = self.Sx.range(ps, q)
                sxx, e_xx = self.Sxx.range(ps, q)
                sxy, e_xy = self.Sxy.range(ps, q)
                sxx_c = sxx - sx * sx / nn
                sxy_c = sxy - sx * sy / nn
                e_sxx = e_xx + 2.0 * np.abs(sx) * e_x / nn + 4.0 * _EPS * np.abs(sxx)
                e_sxy = e_xy + (np.abs(sx) * e_y + np.abs(sy) * e_x) / nn + 4.0 * _EPS * (np.abs(sxy) + np.abs(sx * sy) / nn)
                use = (n >= 3) & (sxx_c > 0)  # fit_poly reduces the order below order + 2 valid samples
                den = np.where(use, sxx_c, 1.0)
                gain = sxy_c * sxy_c / den
                sse = np.where(use, sse - gain, sse)
                err = err + np.where(use, (2.0 * np.abs(sxy_c) * e_sxy + gain * e_sxx) / den + 4.0 * _EPS * gain, 0.0)
                # A slope term that the rounding could create or remove: no reliable fast answer.
                err = np.where(use & (e_sxx > 1e-6 * sxx_c), math.inf, err)
        sse = np.maximum(sse, 0.0)
        # Not resolved above the rounding: fit those segments directly.
        bad = np.flatnonzero(~(sse > 16.0 * err) & (n > 0))
        for i in bad:
            sse[i] = self._direct(self.fast[st], int(ps[i]), int(q))
        return sse

    def cost(self, st: str, p: int, q: int) -> float:
        if self.fast[st] is not None:
            return float(self.vector(st, np.array([p]), q)[0])
        key = (st, p, q)
        c = self.cache.get(key)
        if c is None:
            self.n_fits += 1
            yy = self.y[p:q]
            if not np.isfinite(yy).any():
                c, reduced = 0.0, False  # nothing to explain
            else:
                r = _fit.fit_model(self.models[st], self.ts[p:q], yy, st)
                e = r.residuals[np.isfinite(r.residuals)]
                # A model that cannot be fitted to this segment (too few samples) cannot describe it.
                c = float(e @ e) if len(e) else math.inf
                reduced = bool(set(r.flags) & _REDUCED_FLAGS)
            self.cache[key] = c
            self.reduced[key] = reduced
        return c

    def full_model(self, st: str, p: int, q: int) -> bool:
        """The fit of ``[p, q)`` used the whole model (no order reduction): costs grow from here on."""
        return not self.reduced.get((st, p, q), False)


def _n_params(models: Mapping[str, Any], st: str, ts: np.ndarray, y: np.ndarray) -> int:
    """Free parameters of the state's model, from a probe fit on the first valid samples."""
    ok = np.flatnonzero(np.isfinite(y))[:200]
    if len(ok) < 8:
        return 1
    r = _fit.fit_model(models[st], ts[ok], y[ok], st)
    return max(int(r.n_params), 1)


def _noise_var(y: np.ndarray) -> float:
    """Robust noise variance from first differences: (1.4826 MAD(diff))^2 / 2."""
    d = np.diff(y[np.isfinite(y)])
    if len(d) < 2:
        return 0.0
    mad = float(np.median(np.abs(d - np.median(d))))
    return (1.4826 * mad) ** 2 / 2.0


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class OptimalSegmentation:
    """
    The result of ``optimal_segments``.

    ``segments`` cover the record in order, like ``states.segments``, so
    ``report.fit_report`` and ``metrics.stats_by_state`` take them as they are.
    ``objective = sse + sum of penalties``; ``rms = sqrt(sse / n_valid)``.
    """

    segments: List[Segment]
    method: str
    objective: float
    sse: float
    rms: float
    n_valid: int
    penalty: Dict[str, float]
    counts: Any  # DataFrame: segments touching each period, by state
    n_candidates: int
    n_fits: int = 0
    info: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        n = {st: sum(s.state.value == st for s in self.segments) for st in self.penalty}
        parts = ", ".join(f"{k} {v}" for k, v in n.items())
        return (f"{self.method}: {len(self.segments)} segments ({parts}); RMS {self.rms:.6g}, "
                f"SSE {self.sse:.6g}, objective {self.objective:.6g}; {self.n_candidates} candidate breakpoints")

    def __str__(self) -> str:
        return self.summary()


def compare_segments(a: Sequence[Segment], b: Sequence[Segment], n: Optional[int] = None) -> Dict[str, Any]:
    """
    How far two segmentations of the same record agree, sample by sample:
    ``agreement`` (fraction of samples in the same state), and ``confusion``
    (a DataFrame of sample counts, ``a`` states by ``b`` states). Samples
    that ``b`` leaves out (partial coverage) count as ``"none"``.
    """
    import pandas as pd

    n = n if n is not None else max([s.stop for s in a] + [s.stop for s in b] + [0])

    def labels(segs):
        out = np.full(n, "none", dtype=object)
        for s in check_segments(segs, n):
            out[s.slice] = s.state.value
        return out

    la, lb = labels(a), labels(b)
    conf = pd.crosstab(pd.Series(la, name="a"), pd.Series(lb, name="b"))
    return {"agreement": float(np.mean(la == lb)) if n else float("nan"), "confusion": conf}


# ---------------------------------------------------------------------------
# The dynamic program
# ---------------------------------------------------------------------------


class _Problem:
    def __init__(self, ts, y, idx, models, states, per, period_origin, max_counts, transitions,
                 min_duration_s, max_duration_s, penalty):
        self.ts, self.y, self.idx = ts, y, idx
        self.n = len(ts)
        self.states = list(states)
        self.S = len(self.states)
        self.costs = _Costs(ts, y, models, self.states)
        self.pid = _period_ids(ts, idx, per, period_origin)
        if self.pid is None:
            self.pid = np.zeros(self.n, dtype=np.int64)
        self.dt = float(np.median(np.diff(ts))) if self.n > 1 else 1.0
        self.pen = np.array([penalty[st] for st in self.states], dtype=float)
        self.min_d = np.array([float((min_duration_s or {}).get(st, 0.0)) for st in self.states])
        self.max_d = np.array([float((max_duration_s or {}).get(st, math.inf)) for st in self.states])
        # Transitions: allowed[prev, next].
        if transitions is None:
            self.allowed = np.ones((self.S, self.S), dtype=bool)
        else:
            self.allowed = np.zeros((self.S, self.S), dtype=bool)
            for a, nexts in transitions.items():
                for b in nexts:
                    self.allowed[self.states.index(a), self.states.index(b)] = True
        # Count layers: counts in the current period of the states that are limited.
        self.limited = [i for i, st in enumerate(self.states) if st in (max_counts or {})]
        self.maxc = [int(max_counts[self.states[i]]) for i in self.limited]
        radix = [m + 1 for m in self.maxc]
        self.L = int(np.prod(radix)) if radix else 1
        self.layers = list(np.ndindex(*radix)) if radix else [()]
        index = {c: k for k, c in enumerate(self.layers)}

        def plus(c, s):
            if s not in self.limited:
                return c
            k = self.limited.index(s)
            v = list(c)
            v[k] += 1
            return tuple(v)

        zero = tuple(0 for _ in self.limited)
        self.tgtA = np.full((self.S, self.L), -1, dtype=int)  # same period: c + e_s
        self.tgtB = np.full(self.S, -1, dtype=int)            # fresh period: e_s
        for s in range(self.S):
            e = plus(zero, s)
            if all(v <= m for v, m in zip(e, self.maxc)):
                self.tgtB[s] = index[e]
            for l, c in enumerate(self.layers):
                ce = plus(c, s)
                if all(v <= m for v, m in zip(ce, self.maxc)):
                    self.tgtA[s, l] = index[ce]

    def duration(self, p, q):
        return self.ts[q - 1] - self.ts[p] + self.dt

    def segment_ok(self, s: int, p: int, q: int) -> bool:
        d = self.duration(p, q)
        return self.min_d[s] - 1e-9 <= d <= self.max_d[s] + 1e-9

    def feasible(self, bounds: Sequence[int], sts: Sequence[int]) -> bool:
        """Check a full segmentation (boundaries incl. 0 and n; one state per segment) against the limits."""
        counts: Dict[int, List[int]] = {}
        for i, s in enumerate(sts):
            p, q = bounds[i], bounds[i + 1]
            if not self.segment_ok(s, p, q):
                return False
            if i and not self.allowed[sts[i - 1], s]:
                return False
            if s in self.limited:
                k = self.limited.index(s)
                for per_ in range(int(self.pid[p]), int(self.pid[q - 1]) + 1):
                    c = counts.setdefault(per_, [0] * len(self.limited))
                    c[k] += 1
                    if c[k] > self.maxc[k]:
                        return False
        return True

    def total(self, bounds, sts) -> Tuple[float, float]:
        sse = sum(self.costs.cost(self.states[s], bounds[i], bounds[i + 1]) for i, s in enumerate(sts))
        return sse + float(sum(self.pen[s] for s in sts)), sse

    def solve(self, B: np.ndarray, *, surrogate: bool = False, ub: float = math.inf, prune: bool = True):
        """Optimal segmentation with breakpoints from ``B`` (sorted, with 0 and n). Returns (bounds, states)."""
        m, S, L = len(B), self.S, self.L
        F = np.full((m, S, L), math.inf)
        bp_j = np.full((m, S, L), -1, dtype=np.int64)
        bp_s = np.full((m, S, L), -1, dtype=np.int64)
        bp_l = np.full((m, S, L), -1, dtype=np.int64)
        starts = B[:-1]
        p0 = self.pid[starts]
        prevend = np.concatenate(([np.iinfo(np.int64).min], self.pid[starts[1:] - 1]))
        same_all = p0 == prevend
        t_first = self.ts[starts]
        reachable = np.zeros(m, dtype=bool)
        reachable[0] = True
        for k in range(1, m):
            q = int(B[k])
            p1 = self.pid[q - 1]
            js = np.arange(k)
            dur = self.ts[q - 1] - t_first[:k] + self.dt
            spans = p1 > p0[:k]
            same = same_all[:k]
            for s in range(S):
                st = self.states[s]
                okd = (dur >= self.min_d[s] - 1e-9) & (dur <= self.max_d[s] + 1e-9) & reachable[:k]
                if not okd.any():
                    continue
                if self.costs.is_fast(st):
                    C = np.where(okd, self.costs.vector(st, starts[:k], q), math.inf)
                elif surrogate:  # cheap constant-fit costs, only to find a feasible solution
                    C = np.where(okd, _poly0(self.costs, starts[:k], q), math.inf)
                else:
                    C = np.full(k, math.inf)
                    for j in range(k - 1, -1, -1):  # growing segments: costs never decrease, so stop at the bound
                        if dur[j] > self.max_d[s] + 1e-9:
                            break
                        if not okd[j]:
                            continue
                        c = self.costs.cost(st, int(starts[j]), q)
                        C[j] = c
                        # Too short for the model is infinite but not monotone (a longer segment can be fitted);
                        # only a finite cost above the bound, with the model fitted in full, ends the search.
                        # Costs only grow with the segment once the model is fitted in full: a polynomial
                        # fitted with a reduced order (too few samples) can fit a longer segment better.
                        if prune and math.isfinite(c) and c + self.pen[s] > ub and \
                                self.costs.full_model(st, int(starts[j]), q):
                            break
                C = C + self.pen[s]
                if not np.isfinite(C).any():
                    continue
                best_v = np.full(L, math.inf)
                best = np.full((L, 3), -1, dtype=np.int64)
                # From the start of the record: no previous segment, a fresh period.
                if okd[0] and np.isfinite(C[0]) and self.tgtB[s] >= 0:
                    t_ = self.tgtB[s]
                    if C[0] < best_v[t_]:
                        best_v[t_], best[t_] = C[0], (0, -1, -1)
                # From a previous segment ending at B[j], j >= 1.
                if k > 1:
                    V = F[1:k] + C[1:k, None, None]  # (k - 1, S, L)
                    V[:, ~self.allowed[:, s], :] = math.inf
                    for case, mask in (("A", same[1:] & ~spans[1:]), ("B", ~same[1:]), ("C", same[1:] & spans[1:])):
                        if not mask.any():
                            continue
                        sub = V[mask]
                        arg = sub.argmin(axis=0)
                        vmin = np.take_along_axis(sub, arg[None], axis=0)[0]
                        jidx = np.flatnonzero(mask)[arg] + 1
                        for l in range(L):
                            if case == "A":
                                t_ = self.tgtA[s, l]
                            elif case == "B":
                                t_ = self.tgtB[s]
                            else:
                                t_ = self.tgtB[s] if self.tgtA[s, l] >= 0 else -1
                            if t_ < 0:
                                continue
                            sp = int(np.argmin(vmin[:, l]))
                            v = vmin[sp, l]
                            if v < best_v[t_]:
                                best_v[t_], best[t_] = v, (jidx[sp, l], sp, l)
                F[k, s] = best_v
                bp_j[k, s], bp_s[k, s], bp_l[k, s] = best[:, 0], best[:, 1], best[:, 2]
            reachable[k] = np.isfinite(F[k]).any()
        if not np.isfinite(F[m - 1]).any():
            raise ValueError("no segmentation satisfies the limits (counts, durations, transitions) on these "
                             "candidate breakpoints")
        s, l = np.unravel_index(int(np.argmin(F[m - 1])), (S, L))
        k = m - 1
        bounds, sts = [int(B[k])], []
        while k > 0:
            sts.append(int(s))
            j, ps, pl = int(bp_j[k, s, l]), int(bp_s[k, s, l]), int(bp_l[k, s, l])
            bounds.append(int(B[j]))
            k, s, l = j, ps, pl
        return bounds[::-1], sts[::-1]

    def refine(self, bounds: List[int], sts: List[int], window_s: float, max_sweeps: int = 5):
        """
        Move each internal boundary within ``window_s`` while that lowers the total cost (states fixed).

        Each boundary is searched coarse to fine (about 16 positions per level, strides shrinking eightfold),
        and after the first sweep only boundaries next to one that moved are searched again.
        """
        bounds = list(bounds)
        dirty = set(range(1, len(bounds) - 1))
        for _ in range(max_sweeps):
            if not dirty:
                break
            moved = set()
            for i in sorted(dirty):
                lo, hi, cur = bounds[i - 1], bounds[i + 1], bounds[i]
                a, b = sts[i - 1], sts[i]
                t0 = self.ts[cur]
                left = int(np.searchsorted(self.ts, t0 - window_s - 1e-9, side="left"))
                right = int(np.searchsorted(self.ts, t0 + window_s + 1e-9, side="right"))
                first, last = max(left, lo + 1), min(right, hi) - 1
                if last <= first:
                    continue

                def score(x, i=i, lo=lo, hi=hi, a=a, b=b):
                    if not (self.segment_ok(a, lo, x) and self.segment_ok(b, x, hi)):
                        return math.inf
                    if not self.feasible(bounds[:i] + [x] + bounds[i + 1:], sts):
                        return math.inf
                    return self.costs.cost(self.states[a], lo, x) + self.costs.cost(self.states[b], x, hi)

                seen = {cur: score(cur)}
                best_x = cur
                stride = max(1, (last - first) // 16)
                lo_x, hi_x = first, last
                while True:
                    for x in range(lo_x, hi_x + 1, stride):
                        if x not in seen:
                            seen[x] = score(x)
                    best_x = min(seen, key=seen.get)
                    if stride == 1:
                        break
                    lo_x, hi_x = max(first, best_x - stride), min(last, best_x + stride)
                    stride = max(1, stride // 8)
                if best_x != cur and seen[best_x] < seen[cur] - 1e-12 * max(1.0, abs(seen[cur])):
                    bounds[i] = best_x
                    moved.add(i)
            dirty = {j for i in moved for j in (i - 1, i + 1) if 0 < j < len(bounds) - 1}
        return bounds


def _poly0(costs: _Costs, ps: np.ndarray, q: int) -> np.ndarray:
    """Constant-fit SSE for starts ``ps``: the cheap surrogate used only to find a feasible bound."""
    n = costs.N[q] - costs.N[ps]
    syy, _ = costs.Syy.range(ps, q)
    sy, _ = costs.Sy.range(ps, q)
    with np.errstate(invalid="ignore", divide="ignore"):
        sse = np.where(n > 0, syy - sy * sy / np.where(n > 0, n, 1.0), 0.0)
    return np.maximum(sse, 0.0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _candidates(candidates, n: int) -> np.ndarray:
    if candidates is None:
        return np.array([0, n], dtype=np.int64)
    items = list(candidates)
    if items and isinstance(items[0], Segment):
        pts = [s.start for s in check_segments(items, n)] + [s.stop for s in items]
    else:
        pts = [int(x) for x in items]
    pts = np.unique(np.clip(np.asarray(pts + [0, n], dtype=np.int64), 0, n))
    return pts


def optimal_segments(t, y, models: Mapping[str, Any], *, per=None, max_counts: Optional[Mapping[str, int]] = None,
                     transitions: Optional[Mapping[str, Sequence[str]]] = None,
                     min_duration_s: Optional[Mapping[str, float]] = None,
                     max_duration_s: Optional[Mapping[str, float]] = None,
                     penalty: Union[str, float, Mapping[str, float]] = "bic", method: str = "refine",
                     candidates=None, grid_s: Optional[float] = None, refine_s: Optional[float] = None,
                     period_origin=None, prune: bool = True) -> OptimalSegmentation:
    """
    Best-fit segments of ``y`` under per-period limits (see the module docstring).

    Parameters
    ----------
    t : array of seconds, or DatetimeIndex
        Sample times, increasing. With a DatetimeIndex, ``per`` may be a
        string or Timedelta (``"1D"``, ``"7D"``, ``"12h"``) and periods follow
        its wall clock.
    y : array
        The signal. Non-finite samples are ignored by the fits.
    models : mapping
        ``{state: model spec}`` for the states to use (``"steady"``,
        ``"transient"``, ``"indeterminate"``), e.g. ``{"steady": 0,
        "transient": "exp"}``. Specs as in ``rocksteady.fit``.
    per : str, Timedelta or float (seconds), optional
        The period the counts apply to. None: the whole record.
    max_counts : mapping, optional
        ``{state: maximum segments per period}``. Unlisted states are not limited.
    transitions : mapping, optional
        ``{state: [states that may follow it]}``. Default: any state may follow any.
    min_duration_s, max_duration_s : mapping, optional
        Per-state bounds on segment duration.
    penalty : "bic", float, or mapping
        Added per segment (per state with a mapping). See the module docstring.
    method : "subset", "refine" (default) or "exact"
    candidates : list of Segment, or of sample indices
        Candidate breakpoints: required for "subset" and "refine" (for
        example ``states.segments_from(detector, run_output)``), added to the
        grid for "exact".
    grid_s : float, optional
        For "exact": one candidate per ``grid_s`` seconds. Default: every sample.
    refine_s : float, optional
        For "refine": how far a boundary may move. Default: half the median
        gap between candidates. The detector's ``timing.evaluated_window_s``
        is a natural choice for detector candidates.
    period_origin : optional
        Where periods start (a timestamp, or seconds). Default: the Unix
        epoch, i.e. midnight for days.
    prune : bool
        Drop candidate segments whose own cost exceeds a known feasible
        solution's total (see "Cost" in the module docstring).

    Returns
    -------
    OptimalSegmentation
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    ts, idx = _times(t)
    y = np.asarray(y, dtype=float)
    if ts.shape != y.shape or ts.ndim != 1:
        raise ValueError("t and y must be one-dimensional and of the same length")
    n = len(y)
    if n == 0:
        raise ValueError("empty record")
    if not np.all(np.isfinite(ts)) or (n > 1 and not np.all(np.diff(ts) > 0)):
        raise ValueError("t must be finite and strictly increasing")
    if not isinstance(models, Mapping) or not models:
        raise ValueError("models must be a non-empty mapping {state: model spec}")
    states = list(models)
    for st in states:
        State(st)  # raises for an unknown state
    _fit.validate_models({"y": dict(models)}, states=states)
    for name, mapping in (("max_counts", max_counts), ("min_duration_s", min_duration_s),
                          ("max_duration_s", max_duration_s)):
        for st, v in (mapping or {}).items():
            if st not in states:
                raise ValueError(f"{name} names state {st!r}, which has no model")
            if name == "max_counts" and (isinstance(v, bool) or int(v) != v or v < 0):
                raise ValueError(f"max_counts[{st!r}] must be a non-negative integer, got {v!r}")
            if name != "max_counts" and not (float(v) >= 0):
                raise ValueError(f"{name}[{st!r}] must be >= 0, got {v!r}")
    if transitions is not None:
        for a, nexts in transitions.items():
            for st in [a, *nexts]:
                if st not in states:
                    raise ValueError(f"transitions names state {st!r}, which has no model")
    # Penalty per state.
    if isinstance(penalty, str):
        if penalty != "bic":
            raise ValueError(f"penalty must be 'bic', a number or a mapping, got {penalty!r}")
        var = _noise_var(y)
        n_valid = int(np.isfinite(y).sum())
        pen = {st: var * (_n_params(models, st, ts, y) + 1) * math.log(max(n_valid, 2)) for st in states}
    elif isinstance(penalty, Mapping):
        pen = {st: float(penalty.get(st, 0.0)) for st in states}
    else:
        pen = {st: float(penalty) for st in states}
    if any(not (v >= 0 and math.isfinite(v)) for v in pen.values()):
        raise ValueError(f"penalties must be finite and >= 0, got {pen}")

    prob = _Problem(ts, y, idx, models, states, per, period_origin, max_counts, transitions,
                    min_duration_s, max_duration_s, pen)
    if method in ("subset", "refine"):
        if candidates is None:
            raise ValueError(f"method {method!r} needs candidates (segments or sample indices)")
        B = _candidates(candidates, n)
    else:
        if grid_s is None:
            B = np.arange(n + 1, dtype=np.int64)
        else:
            if not grid_s > 0:
                raise ValueError(f"grid_s must be > 0, got {grid_s!r}")
            cell = np.floor((ts - ts[0]) / float(grid_s)).astype(np.int64)
            B = np.flatnonzero(np.diff(np.concatenate(([cell[0] - 1], cell))) != 0)
        B = np.unique(np.concatenate((B, _candidates(candidates, n))))

    ub = math.inf
    slow = any(not prob.costs.is_fast(st) for st in states)
    if prune and slow:
        # A feasible solution first, with cheap constant-fit costs; its true cost bounds the search.
        sb, ss = prob.solve(B, surrogate=True, prune=False)
        ub = prob.total(sb, ss)[0] * (1 + 1e-9) + 1e-12
    bounds, sts = prob.solve(B, ub=ub, prune=prune)
    if method == "refine":
        if refine_s is None:
            gaps = np.diff(ts[np.minimum(B, n - 1)])
            refine_s = 0.5 * float(np.median(gaps[gaps > 0])) if (gaps > 0).any() else 0.0
        bounds = prob.refine(bounds, sts, float(refine_s))
    objective, sse = prob.total(bounds, sts)
    segs = [Segment(bounds[i], bounds[i + 1], State(states[s])) for i, s in enumerate(sts)]
    n_valid = int(np.isfinite(y).sum())
    counts = count_by_period(segs, idx if idx is not None else ts, per, period_origin=period_origin, states=states)
    return OptimalSegmentation(segs, method, float(objective), float(sse),
                               math.sqrt(sse / n_valid) if n_valid else float("nan"), n_valid, pen, counts,
                               int(len(B)), prob.costs.n_fits, {"upper_bound": ub, "refine_s": refine_s})


__all__ = ["optimal_segments", "OptimalSegmentation", "compare_segments", "count_by_period", "METHODS"]

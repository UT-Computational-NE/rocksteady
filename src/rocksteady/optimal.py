"""
Best-fit segments under constraints: optimal segmentation.

``optimal_segments`` splits a record into segments, gives each a state
(``steady``, ``transient`` or ``indeterminate``), and fits each with the model
for its state. It chooses the breakpoints and states that minimise

    sum over segments of (SSE of the segment's fit + penalty of its state or subtype)

subject to the caller's constraints. That is the best piecewise approximation
of the record, in the RMS sense, that the constraints allow.

This answers a different question from the detector. The detector certifies,
causally, that a signal *is* steady within a limit. This optimiser describes
a finished record as well as possible with a given budget of segments. It will
use the segments it is allowed whether or not the data would pass the
detector's criteria. Its segments are "best-fit segments", not certified
steady states; ``compare_segments`` measures how far they agree with the
detector's (``states.segments_from``).

Constraints
-----------
- ``per`` and ``max_counts``: at most ``max_counts[name]`` segments of a
  state or subtype per period (a day, a week, ...). A state's cap counts
  all of its segments, whatever their subtype. A segment counts once against
  every period it touches, so one segment from 16:00 to 08:00 the next day
  counts against both days. Periods are half-open, and a segment touches a
  period when one of its samples lies in it. For a ``DatetimeIndex``,
  periods follow its wall clock (local days for a local time zone). When
  the clock falls back an hour, the repeated wall time stays in the period
  already in progress, so a period shorter than an hour is not counted
  twice; days are unaffected.
- ``transitions``: which state or subtype may follow which. Default: any.
  This constrains the sequence of segments only. Segments are fitted
  independently, so consecutive fits need not meet: a level change can sit
  at any boundary.
  Alternation forbids steady -> steady, but the optimiser may then place a
  short transient on a plateau next to a step rather than across it.
- ``initial`` / ``final``: the states or subtypes the record's first and
  last segments may take. Default: any. A one-segment record must satisfy
  both. A data gap does not start or end the record.
- ``min_duration_s`` / ``max_duration_s``: per state or subtype. The
  duration of a segment is from its first to its last sample, plus one
  sample spacing. A state whose minimum exceeds its maximum is refused.
- ``penalty``: added for every segment, per state or subtype. With 0, more
  segments never fit worse, so every allowed slot is used. The default,
  ``"bic"``, charges ``sigma^2 (p + 1) ln n`` for a model with ``p`` parameters,
  ``sigma`` estimated from the first differences of the data (robustly), so
  "at most" stays "at most".

When no segmentation satisfies the constraints, the ``ValueError`` names the
earliest candidate time after which none can continue, the period containing
it (its start, as ``count_by_period`` indexes periods, or the whole record),
and the caps and transitions in force (and ``initial`` and ``final``, when
given). It does not search for the constraint to blame.

Subtypes
--------
``subtypes={name: Subtype}`` refines states into caller-named subtypes, such
as cold and hot steady segments, each with conditions on the segment alone:
its level, its peak, its rise time, and whether it contains an event
(``Subtype``). An event condition reads one of the boolean masks passed as
``events={name: mask}``, so that a driven transient (the heater switched)
is told apart from an undriven one, by actuator. Once a state has
subtypes, every segment of it takes one of them (ADR 0001), so no cap can be
dodged by leaving a segment untyped; a subtype with no conditions is a
catch-all. States without subtypes are unchanged, and their segments have
``subtype=None``. A subtype is fitted with its state's model.

Every constraint can name subtypes as well as states. In ``transitions``,
``initial`` and ``final`` a state's name stands for all of its subtypes, and
the allowed pairs or names are the union over all entries, so the operating
sequence ``cold -> heat-up -> hot -> cool-down -> cold`` tells apart two
transient subtypes with the same conditions by their neighbours. A
subtype's durations and its state's both bind: no minimum may exceed a
maximum. A
subtype's penalty replaces its state's; without one (and with ``"bic"``) it
takes its state's.

Subtypes of one state may overlap. The optimiser chooses by cost: a
subtype's segments are fitted with its state's model, so two subtypes that
accept a segment and carry the same penalty tie exactly, and the tie goes
to the subtype declared first. Precisely, among segmentations of equal
objective it takes the least sum, over segments, of each subtype's position
among its state's subtypes, so no segment could, on its own, take an
earlier-declared subtype that accepts it, at the same penalty, without
breaking a cap, a transition, or ``initial`` or ``final``.
That sum also decides exact ties between different states or boundaries,
so declaring subtypes can change which of two equally good segmentations
is returned. ``refine`` keeps every segment's subtype while it moves
boundaries, and rejects a move after which a segment fails its subtype's
conditions. A moved segment can then meet an earlier-declared subtype's
conditions, so ``refine`` types its final segments again with the same
program, their breakpoints and states fixed and every constraint in force:
the tie rule holds for every method.

``classify_segments`` gives subtypes to segments the analyst already has
(the detector's, say) by the same definitions: the same program with the
breakpoints and states fixed and no count caps, durations or penalties, so
ties go to the least sum of positions as above, and a segmentation that
cannot be typed raises an error that names where.

Methods
-------
All three run one dynamic program over candidate breakpoints, which tracks
the last segment's state and the counts used in the current period.

- ``"subset"``: candidates are the boundaries given in ``candidates`` (for
  example the detector's segments). Optimal among those.
- ``"refine"``: ``subset``, then each kept boundary is moved within
  ``refine_s`` to lower the total cost, keeping states and constraints. A local
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
for a strict guarantee, or pass ``prune=False``. Pruning is decided on the
model cost before any subtype's condition, so it stays exact with subtypes:
a longer segment never costs less, whichever subtype could take it.
"""

import math
from dataclasses import dataclass, field, replace
from numbers import Real
from typing import (AbstractSet, Any, Callable, Collection, Dict, FrozenSet, Iterable, List, Mapping, Optional, Protocol,
                    Sequence, Sized, Tuple, Union, cast)

import numpy as np

from . import _validate as _v
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
    ts, idx = _times(t)
    segs = check_segments(segments, len(ts))
    pid = _period_ids(ts, idx, per, period_origin)
    states = list(states) if states is not None else sorted({s.state.value for s in segs})
    return _count_table([(s.start, s.stop, (s.state.value,)) for s in segs], states, pid, idx, per, period_origin)


def _count_table(spans: Sequence[Tuple[int, int, Sequence[str]]], columns: Sequence[str], pid: Optional[np.ndarray],
                 idx, per, period_origin):
    """
    The table of ``count_by_period``. Each span ``(start, stop, names)``
    counts once, under each of its names, against every period it touches.
    """
    import pandas as pd

    if pid is None:
        pid = np.zeros(max([q for _, q, _ in spans] + [0]), dtype=np.int64)
    rows: Dict[int, Dict[str, int]] = {}
    # Periods inside a segment hold no sample of their own only when a gap spans them; they still count.
    for start, stop, names in spans:
        for p in range(int(pid[start]), int(pid[stop - 1]) + 1):
            row = rows.setdefault(p, {})
            for name in names:
                row[name] = row.get(name, 0) + 1
    # A row for each period some column's name touches; every cell an integer count, 0 where it has none.
    periods = np.array(sorted(p for p, row in rows.items() if any(c in row for c in columns)), dtype=np.int64)
    out = pd.DataFrame({c: np.fromiter((rows[p].get(c, 0) for p in periods), np.int64, len(periods)) for c in columns},
                       columns=list(columns))
    out.index = _period_starts(periods, idx, per, period_origin)
    out.index.name = "period"
    return out


def _period_starts(ids: np.ndarray, idx, per, period_origin):
    """The start of each period number in ``ids``, as ``count_by_period`` indexes periods (``"all"`` for none)."""
    import pandas as pd

    if per is None:
        return pd.Index(["all"] * len(ids))
    if idx is not None:
        origin = pd.Timestamp("1970-01-01") if period_origin is None else pd.Timestamp(period_origin).tz_localize(None)
        return pd.Index(origin + ids * pd.Timedelta(per))
    return pd.Index((0.0 if period_origin is None else float(period_origin)) + ids * float(per))


# ---------------------------------------------------------------------------
# Segment costs
# ---------------------------------------------------------------------------


_EPS = float(np.finfo(float).eps)
_REDUCED_FLAGS = frozenset({"order_reduced"})  # the fit used fewer parameters than the model has


_BLOCK = 64  # samples per block of _RangeSum and _RangeMax


def _blocked(x: np.ndarray, size: int, fill: float) -> np.ndarray:
    """``x`` padded with ``fill`` to whole blocks covering indices ``[0, size)``, as (blocks, ``_BLOCK``)."""
    nb = -(-size // _BLOCK)
    v = np.full(nb * _BLOCK, fill)
    v[:len(x)] = x
    return v.reshape(nb, _BLOCK)


class _RangeSum:
    """
    Sums of ``x`` over ``[p, q)`` from prefix sums, with an error bound.

    Prefix sums lose a short range's sum when the record elsewhere is much
    larger. Here the prefix is kept in blocks of ``_BLOCK`` samples: a running sum
    within each block, and the block totals accumulated with compensated
    (Neumaier) summation. The error of a range sum is then bounded by the
    range's own magnitude and the magnitudes of the two blocks at its ends,
    not by the whole record before it. ``range`` returns the sum and that bound.
    """

    def __init__(self, x: np.ndarray):
        blocks = _blocked(x, len(x) + 1, 0.0)  # so that index n is in range
        nb = len(blocks)
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
        bp, bq = p // _BLOCK, q // _BLOCK
        v = (self.hi[bq] - self.hi[bp]) + ((self.lo[bq] - self.lo[bp]) + (self.loc[q] - self.loc[p]))
        err = 4.0 * _EPS * (np.abs(v) + _BLOCK * (self.mass[bp] + self.mass[bq]))
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
            order = int(opts.get("order", 0)) if name == "poly" else None
            self.fast[st] = order if order is not None and order <= 1 else None
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
                e_sxy = (e_xy + (np.abs(sx) * e_y + np.abs(sy) * e_x) / nn
                         + 4.0 * _EPS * (np.abs(sxy) + np.abs(sx * sy) / nn))
                use = (n >= 3) & (sxx_c > 0)  # fit_poly reduces the order below order + 2 valid samples
                den = np.where(use, sxx_c, 1.0)
                gain = sxy_c * sxy_c / den
                sse = np.where(use, sse - gain, sse)
                err = err + np.where(use, (2.0 * np.abs(sxy_c) * e_sxy + gain * e_sxx) / den + 4.0 * _EPS * gain, 0.0)
                # With three or more valid samples (distinct times, as t increases) the slope term exists. If
                # the time variance is not resolved above its rounding, including when it rounds to zero or
                # below (a short segment in a record with a large time gap), there is no reliable fast answer.
                err = np.where((n >= 3) & ~(sxx_c > 1e6 * e_sxx), math.inf, err)
        sse = np.maximum(sse, 0.0)
        # Not resolved above the rounding: fit those segments directly.
        bad = np.flatnonzero(~(sse > 16.0 * err) & (n > 0))
        order = cast(int, self.fast[st])  # vector() takes fast models only
        for i in bad:
            sse[i] = self._direct(order, int(ps[i]), int(q))
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


class _SegmentCosts(Protocol):
    """What the program asks of segment costs: ``_Costs``, or ``_NoCosts`` for a fixed segmentation."""

    n_fits: int

    def is_fast(self, st: str) -> bool: ...

    def vector(self, st: str, ps: np.ndarray, q: int) -> np.ndarray: ...

    def cost(self, st: str, p: int, q: int) -> float: ...

    def full_model(self, st: str, p: int, q: int) -> bool: ...


class _NoCosts:
    """
    The costs of a fixed segmentation (``classify_segments``, and ``refine``'s
    result typed again): every segment costs 0. With the breakpoints and
    states fixed, a segment's model cost is the same whichever of its
    state's subtypes it takes, so only the penalties and then the subtypes'
    ranks decide, as they do between equal objectives in the optimiser.
    """

    n_fits = 0  # nothing to fit

    def is_fast(self, st: str) -> bool:
        return True

    def vector(self, st: str, ps: np.ndarray, q: int) -> np.ndarray:
        return np.zeros(len(ps))

    def cost(self, st: str, p: int, q: int) -> float:
        return 0.0

    def full_model(self, st: str, p: int, q: int) -> bool:
        return True


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
    ``penalty`` is charged per segment: each state's, then each subtype's.
    ``counts`` has a column per state and per declared subtype,
    ``subtypes`` (also ``info["subtypes"]``) holds the subtypes used, and
    ``info["events"]`` the event masks.
    """

    segments: List[Segment]
    method: str
    objective: float
    sse: float
    rms: float
    n_valid: int
    penalty: Dict[str, float]
    counts: Any  # DataFrame: segments touching each period, by state and subtype
    n_candidates: int
    n_fits: int = 0
    info: Dict[str, Any] = field(default_factory=dict)
    subtypes: Dict[str, "Subtype"] = field(default_factory=dict)

    def summary(self) -> str:
        subtypes = self.subtypes
        parts = []
        states = [name for name in self.penalty if name in _STATE_NAMES]  # penalty also holds the subtypes
        for st in states:
            part = f"{st} {sum(s.state.value == st for s in self.segments)}"
            mine = [name for name, sub in subtypes.items() if sub.state.value == st]
            if mine:
                part += ": " + ", ".join(f"{name} {sum(s.subtype == name for s in self.segments)}" for name in mine)
            parts.append(part)
        shape = ("; " if subtypes else ", ").join(parts)
        return (f"{self.method}: {len(self.segments)} segments ({shape}); RMS {self.rms:.6g}, "
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
# Subtypes and their conditions
# ---------------------------------------------------------------------------


_STATE_NAMES = tuple(s.value for s in State)


def _range(value, what: str) -> Tuple[Optional[float], Optional[float]]:
    """A half-open range ``(lo, hi)``, either end None (open); lo < hi when both are given."""
    # Two items in order, each a scalar, in a collection with a length (not an iterator). The items are checked
    # one by one: numpy makes no array of a ragged sequence such as (1, (2, 3)), and older numpy makes an object
    # array of it, with a warning.
    try:
        ordered = isinstance(value, Sized) and not isinstance(value, (str, bytes, Mapping, AbstractSet))
        items = list(value) if ordered else None
    except TypeError:  # a 0-d array has a length method but no length
        items = None
    if items is None or len(items) != 2 or any(np.ndim(v) != 0 for v in items):
        raise ValueError(f"{what} must be a pair (lo, hi), got {value!r}")
    lo, hi = (_v.optional_finite(v, f"{what} {end}") for v, end in zip(items, ("lo", "hi")))
    _v.ordered(lo, hi, f"{what} lo", f"{what} hi", strict=True)
    return lo, hi


@dataclass(frozen=True)
class Subtype:
    """
    A caller-named refinement of one state, for ``optimal_segments``: for
    example a cold or a hot steady segment.

    ``state`` is the state it refines (``"steady"``, ``"transient"`` or
    ``"indeterminate"``). Its conditions, on the segment's finite samples
    except for ``events``:

    - ``level=(lo, hi)``: the mean lies in ``[lo, hi)``. ``None`` leaves an
      end open, so ``(None, 40)`` and ``(40, None)`` split every level at 40
      with no gap and no overlap.
    - ``peak=(lo, hi)``: the maximum lies in ``[lo, hi)``, either end open,
      so that, say, a startup must reach operating level.
    - ``max_rise_s`` with ``rise_fraction`` (default 0.95): with ``y0`` the
      first finite sample and ``top`` the maximum, the time from the
      segment's first sample to its first sample with
      ``y >= y0 + rise_fraction * (top - y0)`` is at most ``max_rise_s``
      seconds. Measured from the segment's own start, it works for a signal
      whose baseline is not zero, and tells a pulse from an ordinary rise.
      The maximum itself always counts as reached.
    - ``events={mask name: True or False}``: the segment does, or does not,
      contain a flagged sample of that mask (one of the ``events`` given to
      ``optimal_segments``), that is, a flagged sample in ``[start, stop)``.
      There is no tolerance: to accept an event just before a segment's
      start, widen the mask. This condition reads the mask alone, whether
      or not the signal is finite there. It is held as ``(mask name, bool)``
      pairs sorted by name, so that a subtype stays hashable plain data;
      ``dict(sub.events)`` gives the mapping back. Such pairs are accepted
      too, so ``dataclasses.replace`` rebuilds a subtype.

    A segment must meet all of its subtype's conditions. A segment with no
    finite sample meets no level, peak or rise condition. A subtype with no
    conditions is a catch-all. A subtype has no model of its own: its
    segments are fitted with its state's.
    """

    state: State
    level: Optional[Tuple[Optional[float], Optional[float]]] = None
    peak: Optional[Tuple[Optional[float], Optional[float]]] = None
    max_rise_s: Optional[float] = None
    rise_fraction: Optional[float] = None
    # Given as {mask name: bool} (or such pairs); held as sorted (name, bool) pairs.
    events: Optional[Union[Mapping[str, bool], Tuple[Tuple[str, bool], ...]]] = None

    def __post_init__(self):
        if self.state not in _STATE_NAMES:
            raise ValueError(f"Subtype state must be one of {', '.join(_STATE_NAMES)}; got {self.state!r}")
        object.__setattr__(self, "state", State(self.state))
        if self.level is not None:
            object.__setattr__(self, "level", _range(self.level, "Subtype level"))
        if self.peak is not None:
            object.__setattr__(self, "peak", _range(self.peak, "Subtype peak"))
        if self.events is not None:
            object.__setattr__(self, "events", _event_conditions(self.events))
        if self.max_rise_s is None:
            if self.rise_fraction is not None:
                raise ValueError("Subtype rise_fraction needs max_rise_s; give both, or neither")
            return
        object.__setattr__(self, "max_rise_s", _v.finite(self.max_rise_s, "Subtype max_rise_s", nonnegative=True))
        f = 0.95 if self.rise_fraction is None else self.rise_fraction
        if isinstance(f, bool) or not isinstance(f, Real) or not 0 < f <= 1:
            raise ValueError(f"Subtype rise_fraction must be in (0, 1], got {self.rise_fraction!r}")
        object.__setattr__(self, "rise_fraction", float(f))


def _event_conditions(events) -> Optional[Tuple[Tuple[str, bool], ...]]:
    """
    ``Subtype.events`` as ``(mask name, bool)`` pairs sorted by name, or None
    when it names no mask. It takes a mapping or such pairs, its own output
    included, so that ``dataclasses.replace`` rebuilds a subtype.
    """
    if isinstance(events, Mapping):
        items = list(events.items())
    else:
        ok = isinstance(events, Iterable) and not isinstance(events, (str, bytes))
        items = list(events) if ok else []
        if not ok or not all(isinstance(v, (tuple, list)) and len(v) == 2 for v in items):
            raise ValueError(f"Subtype events must be a mapping {{mask name: True or False}} or (mask name, True or "
                             f"False) pairs, got {events!r}")
    out = {}
    for name, want in items:
        _v.name(name, "a Subtype events mask name")
        if not isinstance(want, (bool, np.bool_)):
            raise ValueError(f"Subtype events[{name!r}] must be True or False, got {want!r}")
        if name in out:  # only pairs can repeat a name; which one won would depend on their order
            raise ValueError(f"Subtype events names mask {name!r} twice; give each mask once")
        out[name] = bool(want)
    return tuple(sorted(out.items())) if out else None


def _event_pairs(sub: "Subtype") -> Tuple[Tuple[str, bool], ...]:
    """A subtype's event conditions as ``Subtype`` holds them: ``(mask name, bool)`` pairs."""
    return cast(Tuple[Tuple[str, bool], ...], sub.events or ())


def _event_masks(events, n: int, subtypes: Mapping[str, "Subtype"]) -> Dict[str, np.ndarray]:
    """
    The ``events`` of ``optimal_segments`` as boolean arrays, one value per
    sample: every mask is checked, and every mask a condition names must be
    given.
    """
    if events is not None and not isinstance(events, Mapping):
        raise ValueError("events must be a mapping {name: boolean mask}; give a single mask a name, "
                         "e.g. events={'heater': mask}")
    masks = {}
    for name, mask in (events or {}).items():
        _v.name(name, "an event mask's name")
        m = np.array(mask)  # a copy: the result keeps the masks it used
        if m.shape != (n,):
            raise ValueError(f"events[{name!r}] must be a one-dimensional mask with one value per sample of t ({n}), "
                             f"got shape {m.shape}")
        if m.dtype != bool:
            import pandas as pd

            missing = int(pd.isna(m).sum())
            if missing:
                raise ValueError(f"events[{name!r}] must be boolean with no missing values, got {missing} missing")
            raise ValueError(f"events[{name!r}] must be boolean, got values of type {m.dtype}")
        masks[name] = m
    for name, sub in subtypes.items():
        for mask, _ in _event_pairs(sub):
            if mask not in masks:
                raise ValueError(f"subtype {name!r} needs event mask {mask!r}, which events does not include")
    return masks


def _as_integers(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Finite floats ``x`` exactly as ``m * 2**e``: the integers ``m`` and ``e`` (``m = 0`` for zero)."""
    mant, e = np.frexp(x)
    return (mant * 2.0 ** 53).astype(np.int64), e.astype(np.int64) - 53  # a mantissa has 53 bits


class _Levels:
    """
    The level condition, vectorised over segment starts. Means come from
    range sums with an error bound; where a mean is too close to an end of
    the range to decide from them, the segment is compared exactly
    (``at_least``), so ``[lo, hi)`` holds for the exact mean.
    """

    def __init__(self, y: np.ndarray):
        ok = np.isfinite(y)
        self.ok = ok
        self.y = y
        self.N = np.concatenate(([0.0], np.cumsum(ok.astype(float))))
        self.S = _RangeSum(np.where(ok, y, 0.0))
        self._exact: Optional[Tuple[np.ndarray, int]] = None

    @property
    def exact_prefix(self) -> Tuple[np.ndarray, int]:
        """
        Exact prefix sums of the finite samples, ``(P, e0)``: the sum over
        ``[p, q)`` is ``(P[q] - P[p]) * 2**e0``, with ``P`` Python integers.
        Built on the first close comparison. ``e0`` is the smallest exponent
        of any nonzero sample, so the integers grow with the record's range
        of exponents: one subnormal sample among values near 1 makes them
        about 1100 bits wide.
        """
        if self._exact is None:
            m, e = _as_integers(np.where(self.ok, self.y, 0.0))
            nonzero = m != 0
            e0 = int(e[nonzero].min()) if nonzero.any() else 0
            P = np.zeros(len(m) + 1, dtype=object)
            acc = 0
            for i, (mi, ei) in enumerate(zip(m.tolist(), np.where(nonzero, e - e0, 0).tolist())):
                acc += mi << ei
                P[i + 1] = acc
            self._exact = P, e0
        return self._exact

    def at_least(self, ps: np.ndarray, q: int, thr: float) -> np.ndarray:
        """
        Whether the mean of the finite samples of ``[p, q)`` is at least ``thr``, exactly, for each start in
        ``ps`` (each range with a finite sample). The sums are exact integers, so no rounding can decide:
        ``fsum(y) / n`` rounds twice, and 53 samples of 0.7 would fall below 0.7.
        """
        P, e0 = self.exact_prefix
        (mt,), (et,) = (v.tolist() for v in _as_integers(np.array([thr])))
        total = P[q] - P[ps]  # times 2**e0
        target = (self.N[q] - self.N[ps]).astype(np.int64).astype(object) * mt  # times 2**et
        if e0 >= et:
            return np.asarray(total * (1 << (e0 - et)) >= target, dtype=bool)
        return np.asarray(total >= target * (1 << (et - e0)), dtype=bool)

    def accepts(self, lo: Optional[float], hi: Optional[float]) -> Callable[[np.ndarray, int], np.ndarray]:
        def level_ok(ps: np.ndarray, q: int) -> np.ndarray:
            ps = np.asarray(ps, dtype=np.int64)
            n = self.N[q] - self.N[ps]
            s, e = self.S.range(ps, q)
            nn = np.where(n > 0, n, 1.0)
            mean = s / nn
            err = e / nn + 4.0 * _EPS * np.abs(mean)
            out = n > 0
            near = np.zeros(len(ps), dtype=bool)
            for thr, above in ((lo, True), (hi, False)):
                if thr is None:
                    continue
                out &= (mean >= thr) if above else (mean < thr)
                near |= np.abs(mean - thr) <= err
            close = np.flatnonzero(near & (n > 0))
            if len(close):  # decided exactly, all at once
                within = np.ones(len(close), dtype=bool)
                if lo is not None:
                    within &= self.at_least(ps[close], q, lo)
                if hi is not None:
                    within &= ~self.at_least(ps[close], q, hi)
                out[close] = within
            return out

        return level_ok


class _RangeMax:
    """
    The maximum of the finite samples of ``y`` over ``[p, q)``, vectorised
    over ranges (``-inf`` for a range with no finite sample); exact, as a
    maximum is. Like ``_RangeSum`` it works in blocks of ``_BLOCK`` samples: the
    running maxima from each block's start and to its end, and a sparse
    table over whole blocks, so it takes about three values per sample. A
    range inside one block is read directly.
    """

    def __init__(self, y: np.ndarray):
        blocks = _blocked(np.where(np.isfinite(y), y, -math.inf), len(y), -math.inf)
        nb = len(blocks)
        self.v = blocks.ravel()
        self.pre = np.maximum.accumulate(blocks, axis=1).ravel()  # from the block's start to i
        self.suf = np.maximum.accumulate(blocks[:, ::-1], axis=1)[:, ::-1].ravel()  # from i to the block's end
        self.table = [blocks.max(axis=1)]  # table[k][b]: the maximum of blocks [b, b + 2**k)
        while 2 ** len(self.table) <= nb:
            prev, h = self.table[-1], 2 ** (len(self.table) - 1)
            self.table.append(np.maximum(prev[:-h], prev[h:]))

    def _blocks(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """The maximum of whole blocks ``[a, b)``; ``-inf`` where the range is empty."""
        out = np.full(len(a), -math.inf)
        size = b - a
        k = np.frexp(np.maximum(size, 1))[1] - 1  # floor(log2(size))
        for level in np.unique(k[size > 0]):
            sel = (size > 0) & (k == level)
            t = self.table[int(level)]
            out[sel] = np.maximum(t[a[sel]], t[b[sel] - 2 ** int(level)])
        return out

    def range(self, ps, qs) -> np.ndarray:
        ps, qs = np.broadcast_arrays(np.asarray(ps, dtype=np.int64), np.asarray(qs, dtype=np.int64))
        bp, bq = ps // _BLOCK, (qs - 1) // _BLOCK
        out = np.full(len(ps), -math.inf)
        apart = bp < bq
        if apart.any():
            p, q = ps[apart], qs[apart]
            out[apart] = np.maximum(np.maximum(self.suf[p], self.pre[q - 1]), self._blocks(bp[apart] + 1, bq[apart]))
        within = ~apart & (qs > ps)
        if within.any():
            p, q = ps[within], qs[within]
            ix = p[:, None] + np.arange(_BLOCK)
            out[within] = np.where(ix < q[:, None], self.v[np.minimum(ix, len(self.v) - 1)], -math.inf).max(axis=1)
        return out


class _Conditions:
    """
    Every subtype's conditions as one test of candidate segments,
    ``accepts(ps, q)``. The sums, maxima and counts of flagged samples the
    conditions need are built on first use.
    """

    def __init__(self, ts: np.ndarray, y: np.ndarray, events: Optional[Mapping[str, np.ndarray]] = None):
        self.ts, self.y = ts, y
        self.events = dict(events or {})
        self._flags: Dict[str, np.ndarray] = {}  # flagged samples of each mask before each index
        self._levels: Optional[_Levels] = None
        self._max: Optional[_RangeMax] = None
        self._next: Optional[np.ndarray] = None

    @property
    def maxima(self) -> _RangeMax:
        if self._max is None:
            self._max = _RangeMax(self.y)
        return self._max

    @property
    def next_finite(self) -> np.ndarray:
        """For each sample, the index of the first finite sample at or after it (``n`` for none)."""
        if self._next is None:
            n = len(self.y)
            ix = np.where(np.isfinite(self.y), np.arange(n), n)
            self._next = np.minimum.accumulate(ix[::-1])[::-1]
        return self._next

    def level(self, lo, hi) -> Callable[[np.ndarray, int], np.ndarray]:
        if self._levels is None:
            self._levels = _Levels(self.y)
        return self._levels.accepts(lo, hi)

    def peak(self, lo, hi) -> Callable[[np.ndarray, int], np.ndarray]:
        def peak_ok(ps: np.ndarray, q: int) -> np.ndarray:
            top = self.maxima.range(ps, q)
            return (top > -math.inf) & (lo is None or top >= lo) & (hi is None or top < hi)

        return peak_ok

    def rise(self, max_s: float, fraction: float) -> Callable[[np.ndarray, int], np.ndarray]:
        ts, y, n = self.ts, self.y, len(self.y)

        def rise_ok(ps: np.ndarray, q: int) -> np.ndarray:
            ps = np.asarray(ps, dtype=np.int64)
            first = self.next_finite[ps]
            found = first < q
            top = self.maxima.range(ps, q)
            y0 = y[np.minimum(first, n - 1)]
            with np.errstate(invalid="ignore", over="ignore"):
                thr = np.minimum(y0 + fraction * (top - y0), top)  # the maximum always counts as reached
            # The samples within max_s of the segment's first, as the difference of their times says: [p, end).
            t0 = ts[ps]
            end = np.searchsorted(ts, t0 + max_s, side="right")
            while True:
                back = (end > ps + 1) & (ts[end - 1] - t0 > max_s)
                if not back.any():
                    break
                end -= back
            while True:
                fwd = (end < n) & (ts[np.minimum(end, n - 1)] - t0 <= max_s)
                if not fwd.any():
                    break
                end += fwd
            reached = self.maxima.range(ps, np.minimum(end, q))
            return found & (reached >= thr)

        return rise_ok

    def event(self, name: str, want: bool) -> Callable[[np.ndarray, int], np.ndarray]:
        if name not in self._flags:
            self._flags[name] = np.concatenate(([0], np.cumsum(self.events[name], dtype=np.int64)))
        flags = self._flags[name]

        def event_ok(ps: np.ndarray, q: int) -> np.ndarray:
            return (flags[q] - flags[np.asarray(ps, dtype=np.int64)] > 0) == want

        return event_ok

    def of(self, sub: "Subtype") -> Callable[[np.ndarray, int], np.ndarray]:
        """The test of ``sub``'s conditions: each is checked only on the starts the earlier ones accept."""
        tests = []
        if sub.level is not None:
            tests.append(self.level(*sub.level))
        if sub.peak is not None:
            tests.append(self.peak(*sub.peak))
        if sub.max_rise_s is not None and sub.rise_fraction is not None:  # a Subtype sets both or neither
            tests.append(self.rise(sub.max_rise_s, sub.rise_fraction))
        tests += [self.event(name, want) for name, want in _event_pairs(sub)]
        if not tests:
            return _accept_all
        if len(tests) == 1:
            return tests[0]

        def all_ok(ps: np.ndarray, q: int) -> np.ndarray:
            ps = np.asarray(ps, dtype=np.int64)
            out = tests[0](ps, q)
            for test in tests[1:]:
                keep = np.flatnonzero(out)
                if not len(keep):
                    break
                out[keep] = test(ps[keep], q)
            return out

        return all_ok


# ---------------------------------------------------------------------------
# The dynamic program
# ---------------------------------------------------------------------------


def _accept_all(ps: np.ndarray, q: int) -> np.ndarray:
    return np.ones(len(ps), dtype=bool)


@dataclass(frozen=True)
class _Option:
    """
    One choice the program can give a segment: a state with a model, the
    segment's penalty and duration bounds, and a condition on the segment.
    ``accepts(ps, q)`` says, for each start in ``ps``, whether ``[p, q)``
    meets the condition. ``subtype`` names the choice's subtype (None for a
    state without subtypes), and ``rank`` is its position among its state's
    subtypes, which breaks exact ties in cost.
    """

    state: str
    penalty: float
    min_d: float = 0.0
    max_d: float = math.inf
    accepts: Callable[[np.ndarray, int], np.ndarray] = _accept_all
    subtype: Optional[str] = None
    rank: int = 0


def _fixed_to(segments: Sequence[Segment], opt: _Option) -> Callable[[np.ndarray, int], np.ndarray]:
    """
    ``opt``'s test restricted to the given segments: ``[p, q)`` must be one
    of them, in ``opt``'s state, and meet ``opt``'s condition.
    """
    by_stop = {s.stop: (s.start, State(s.state).value) for s in segments}

    def accepts(ps: np.ndarray, q: int) -> np.ndarray:
        ps = np.asarray(ps, dtype=np.int64)
        p, st = by_stop.get(q, (-1, None))
        out = (ps == p) & (st == opt.state)
        keep = np.flatnonzero(out)
        if len(keep) and opt.accepts is not _accept_all:
            out[keep] = opt.accepts(ps[keep], q)
        return out

    return accepts


@dataclass(frozen=True)
class _Counter:
    """A count cap: at most ``cap`` segments per period whose option is in ``options``."""

    options: FrozenSet[int]
    cap: int


def _lexargmin(v: np.ndarray, r: np.ndarray):
    """
    Along the first axis: the index of the least ``v``, of those the least ``r``, then the first; and the
    ``v`` and ``r`` there. With ``r`` all equal it is ``argmin(v)``.
    """
    vmin = v.min(axis=0)
    arg = np.where(v == vmin, r, np.iinfo(np.int64).max).argmin(axis=0)
    return arg, vmin, np.take_along_axis(r, np.expand_dims(arg, 0), axis=0)[0]


class _Problem:
    def __init__(self, ts, y, idx, models, options: Sequence[_Option], counters: Sequence[_Counter],
                 allowed: np.ndarray, may_start: np.ndarray, may_end: np.ndarray,
                 columns: Sequence[Tuple[str, FrozenSet[int]]], per, period_origin,
                 max_counts: Optional[Mapping[str, int]] = None,
                 transitions: Optional[Mapping[str, Sequence[str]]] = None,
                 initial_final: Optional[Mapping[str, Sequence[str]]] = None,
                 fixed_segments: Optional[Sequence[Segment]] = None):
        self.ts, self.y, self.idx = ts, y, idx
        # Segments covering the record, whose breakpoints and states the program keeps: it only types them.
        self.fixed_segments = fixed_segments
        if fixed_segments is not None:
            options = [replace(opt, accepts=_fixed_to(fixed_segments, opt)) for opt in options]
        # The caller's caps and transitions, as given: the infeasibility error names them.
        self.given_caps, self.given_transitions = max_counts, transitions
        self.given_initial_final = dict(initial_final or {})  # when given
        self.per, self.period_origin = per, period_origin
        self.n = len(ts)
        self.options = list(options)
        self.O = len(self.options)
        self.allowed = np.asarray(allowed, dtype=bool)  # allowed[previous option, next option]
        self.may_start = np.asarray(may_start, dtype=bool)  # may_start[option]: may take the first segment
        self.may_end = np.asarray(may_end, dtype=bool)  # may_end[option]: may take the last segment
        self.columns = list(columns)  # the result's count columns: (name, options counted)
        # The options of each state: one cost per state, shared by its options.
        self.by_state: Dict[str, List[int]] = {}
        for o, opt in enumerate(self.options):
            self.by_state.setdefault(opt.state, []).append(o)
        self.costs: _SegmentCosts = _NoCosts() if fixed_segments is not None else _Costs(ts, y, models, list(self.by_state))
        # Over a state's options: the longest duration allowed (where the cost scan stops), and the smallest
        # penalty (so the pruning bound holds for every option).
        self.state_max_d = {st: max(options[o].max_d for o in members) for st, members in self.by_state.items()}
        self.state_min_pen = {st: min(options[o].penalty for o in members) for st, members in self.by_state.items()}
        pid = _period_ids(ts, idx, per, period_origin)
        self.pid: np.ndarray = pid if pid is not None else np.zeros(self.n, dtype=np.int64)
        self.dt = float(np.median(np.diff(ts))) if self.n > 1 else 1.0
        self.pen = np.array([opt.penalty for opt in self.options], dtype=float)
        self.rank = np.array([opt.rank for opt in self.options], dtype=np.int64)
        self.min_d = np.array([opt.min_d for opt in self.options], dtype=float)
        self.max_d = np.array([opt.max_d for opt in self.options], dtype=float)
        # Count layers: the value of every counter in the current period. A segment adds one to each counter
        # whose set holds its option.
        self.counters = list(counters)
        self.maxc = [c.cap for c in self.counters]
        self.counted_by = [[k for k, c in enumerate(self.counters) if o in c.options] for o in range(self.O)]
        radix = [m + 1 for m in self.maxc]
        self.L = int(np.prod(radix)) if radix else 1
        self.layers = list(np.ndindex(*radix)) if radix else [()]
        index = {c: k for k, c in enumerate(self.layers)}

        def plus(c, o):
            v = list(c)
            for k in self.counted_by[o]:
                v[k] += 1
            return tuple(v)

        zero = tuple(0 for _ in self.counters)
        self.tgtA = np.full((self.O, self.L), -1, dtype=int)  # same period: c + e_o
        self.tgtB = np.full(self.O, -1, dtype=int)            # fresh period: e_o
        for o in range(self.O):
            e = plus(zero, o)
            if all(v <= m for v, m in zip(e, self.maxc)):
                self.tgtB[o] = index[e]
            for l, c in enumerate(self.layers):
                ce = plus(c, o)
                if all(v <= m for v, m in zip(ce, self.maxc)):
                    self.tgtA[o, l] = index[ce]
        # The options of each state that can be placed at all. One that alone exceeds a cap (a cap of 0) never
        # can, so its segments are never costed.
        self.placeable = {st: [o for o in members if self.tgtB[o] >= 0] for st, members in self.by_state.items()}

    def duration(self, p, q):
        return self.ts[q - 1] - self.ts[p] + self.dt

    def segment_ok(self, o: int, p: int, q: int) -> bool:
        d = self.duration(p, q)
        if not self.min_d[o] - 1e-9 <= d <= self.max_d[o] + 1e-9:
            return False
        return bool(self.options[o].accepts(np.array([p]), q)[0])

    def cost(self, o: int, p: int, q: int) -> float:
        return self.costs.cost(self.options[o].state, p, q)

    def feasible(self, bounds: Sequence[int], opts: Sequence[int]) -> bool:
        """Check a full segmentation (boundaries incl. 0 and n; one option per segment) against the constraints."""
        if not (self.may_start[opts[0]] and self.may_end[opts[-1]]):
            return False
        counts: Dict[int, List[int]] = {}
        for i, o in enumerate(opts):
            p, q = bounds[i], bounds[i + 1]
            if not self.segment_ok(o, p, q):
                return False
            if i and not self.allowed[opts[i - 1], o]:
                return False
            for per_ in range(int(self.pid[p]), int(self.pid[q - 1]) + 1):
                c = counts.setdefault(per_, [0] * len(self.counters))
                for k in self.counted_by[o]:
                    c[k] += 1
                    if c[k] > self.maxc[k]:
                        return False
        return True

    def total(self, bounds, opts) -> Tuple[float, float]:
        sse = sum(self.cost(o, bounds[i], bounds[i + 1]) for i, o in enumerate(opts))
        return sse + float(sum(self.pen[o] for o in opts)), sse

    def counts(self, bounds, opts):
        """Segments touching each period, by count column."""
        names = [tuple(name for name, members in self.columns if o in members) for o in opts]
        spans = [(bounds[i], bounds[i + 1], names[i]) for i in range(len(opts))]
        return _count_table(spans, [name for name, _ in self.columns], self.pid, self.idx, self.per,
                            self.period_origin)

    def _state_costs(self, st: str, okd: np.ndarray, starts: np.ndarray, q: int, dur: np.ndarray, surrogate: bool,
                     ub: float, prune: bool) -> np.ndarray:
        """Model costs of ``[starts[j], q)`` for state ``st`` where ``okd``; infinite elsewhere."""
        k = len(okd)
        if self.costs.is_fast(st):
            return np.where(okd, self.costs.vector(st, starts[:k], q), math.inf)
        if surrogate:  # cheap constant-fit costs, only to find a feasible solution
            # Only a model that is not fast has a surrogate, and only _Costs has such models.
            return np.where(okd, _poly0(cast(_Costs, self.costs), starts[:k], q), math.inf)
        C = np.full(k, math.inf)
        max_d, min_pen = self.state_max_d[st], self.state_min_pen[st]
        for j in range(k - 1, -1, -1):  # growing segments: costs never decrease, so stop at the bound
            if dur[j] > max_d + 1e-9:
                break
            if not okd[j]:
                continue
            c = self.costs.cost(st, int(starts[j]), q)
            C[j] = c
            # Too short for the model is infinite but not monotone (a longer segment can be fitted);
            # only a finite cost above the bound, with the model fitted in full, ends the search.
            # Costs only grow with the segment once the model is fitted in full: a polynomial
            # fitted with a reduced order (too few samples) can fit a longer segment better.
            # The bound uses the model cost alone, before any condition, so pruning stays exact.
            if prune and math.isfinite(c) and c + min_pen > ub and self.costs.full_model(st, int(starts[j]), q):
                break
        return C

    def solve(self, B: np.ndarray, *, ub: float = math.inf, prune: bool = True):
        """Optimal segmentation with breakpoints from ``B`` (sorted, with 0 and n). Returns (bounds, options)."""
        table, k = self._forward(B, surrogate=False, ub=ub, prune=prune)
        if table is None:
            raise ValueError(self._infeasible(B, k))
        return self._backtrack(B, *table)

    def pruning_bound_or_raise(self, B: np.ndarray) -> float:
        """
        A bound on the optimal objective for pruning, or the infeasibility
        ``ValueError`` when no segmentation exists: the true cost of the
        optimal segmentation under cheap constant-fit costs. Every segment a
        model can fit has a finite constant fit, so that is a relaxation: when
        it fails, so does the true problem, which gets stuck no later. The
        error then needs the true costs only on the candidates up to there.
        """
        table, k = self._forward(B, surrogate=True, ub=math.inf, prune=False)
        if table is None:
            _, k = self._forward(B[: k + 1], surrogate=False, ub=math.inf, prune=False)
            raise ValueError(self._infeasible(B, k))
        bounds, opts = self._backtrack(B, *table)
        return self.total(bounds, opts)[0] * (1 + 1e-9) + 1e-12

    def _forward(self, B: np.ndarray, *, surrogate: bool, ub: float, prune: bool):
        """
        The program's table over breakpoints ``B``, ``(F, R, bp_j, bp_o, bp_l)``
        (None when no segmentation reaches ``B[-1]``), and the index of the
        latest candidate a segmentation reaches.
        """
        m, O, L = len(B), self.O, self.L
        F = np.full((m, O, L), math.inf)
        R = np.zeros((m, O, L), dtype=np.int64)  # sum of the subtypes' ranks: breaks exact ties in F
        bp_j = np.full((m, O, L), -1, dtype=np.int64)
        bp_o = np.full((m, O, L), -1, dtype=np.int64)
        bp_l = np.full((m, O, L), -1, dtype=np.int64)
        starts = B[:-1]
        p0 = self.pid[starts]
        prevend = np.concatenate(([np.iinfo(np.int64).min], self.pid[starts[1:] - 1]))
        same_all = p0 == prevend
        t_first = self.ts[starts]
        reachable = np.zeros(m, dtype=bool)
        reachable[0] = True
        for k in range(1, m):
            q = int(B[k])
            ends_record = q == self.n  # a segment ending here is the last: its option must be allowed last
            p1 = self.pid[q - 1]
            dur = self.ts[q - 1] - t_first[:k] + self.dt
            spans = p1 > p0[:k]
            same = same_all[:k]
            for st, members in self.placeable.items():
                if ends_record:
                    members = [o for o in members if self.may_end[o]]
                if not members:
                    continue
                okds = [(dur >= self.min_d[o] - 1e-9) & (dur <= self.max_d[o] + 1e-9) & reachable[:k]
                        for o in members]
                okd_any = okds[0] if len(okds) == 1 else np.any(okds, axis=0)
                if not okd_any.any():
                    continue
                C_st = self._state_costs(st, okd_any, starts, q, dur, surrogate, ub, prune)
                for o, okd in zip(members, okds):
                    if not okd.any():
                        continue
                    # The condition acts after the model cost has been used for pruning.
                    accepts = self.options[o].accepts
                    if accepts is not _accept_all:
                        okd = okd & accepts(starts[:k], q)
                    C = np.where(okd, C_st, math.inf) + self.pen[o]
                    if not np.isfinite(C).any():
                        continue
                    F[k, o], R[k, o], bp_j[k, o], bp_o[k, o], bp_l[k, o] = self._best_ending_at(
                        k, o, C, F, R, same, spans)
            reachable[k] = np.isfinite(F[k]).any()
        if not np.isfinite(F[m - 1]).any():
            return None, int(np.max(np.flatnonzero(reachable)))
        return (F, R, bp_j, bp_o, bp_l), m - 1

    def _backtrack(self, B: np.ndarray, F: np.ndarray, R: np.ndarray, bp_j: np.ndarray, bp_o: np.ndarray,
                   bp_l: np.ndarray):
        """The optimal (bounds, options) from the table of ``_forward``: of equal values, the lower rank sum."""
        m = len(B)
        best = int(_lexargmin(F[m - 1].ravel(), R[m - 1].ravel())[0])
        o, l = (int(v) for v in np.unravel_index(best, F.shape[1:]))
        k = m - 1
        bounds, opts = [int(B[k])], []
        while k > 0:
            opts.append(int(o))
            j, po, pl = int(bp_j[k, o, l]), int(bp_o[k, o, l]), int(bp_l[k, o, l])
            bounds.append(int(B[j]))
            k, o, l = j, po, pl
        return bounds[::-1], opts[::-1]

    def _infeasible(self, B: np.ndarray, k: int) -> str:
        """
        The error for a record no segmentation covers. ``B[k]`` is the latest
        candidate breakpoint any segmentation reaches, the earliest candidate
        time after which none can continue. No search for the constraint to blame.
        """
        stuck = int(B[k])
        unit = "" if self.idx is not None else " s"
        at = self.idx[stuck] if self.idx is not None else float(self.ts[stuck])
        if self.per is None:
            where = "the whole record"
        else:
            start = _period_starts(np.array([self.pid[stuck]]), self.idx, self.per, self.period_origin)[0]
            where = f"the period starting {start}{unit}"
        caps = {st: int(v) for st, v in (self.given_caps or {}).items()} or "none"
        if self.per is not None and self.given_caps:
            caps = f"{caps} per period ({self.per})"
        given = self.given_transitions
        trans = "any" if given is None else {a: sorted(set(b)) for a, b in given.items()}
        initial_final = "".join(f", {what}={names}" for what, names in self.given_initial_final.items())
        if self.fixed_segments is not None:  # classify_segments: no counts or durations; the caller's segments
            return (f"these segments cannot be typed: no choice of subtypes satisfies the transitions, first and "
                    f"last segments and the subtypes' conditions: none can continue from {at}{unit} (sample "
                    f"{stuck}), in {where}. In force: transitions={trans}{initial_final}.")
        return (f"no segmentation satisfies the constraints (counts, durations, transitions, first and last "
                f"segments) and the subtypes' conditions on these candidate breakpoints: none can continue from "
                f"{at}{unit} (sample {stuck}), in {where}. In force: max_counts={caps}, "
                f"transitions={trans}{initial_final}.")

    def _best_ending_at(self, k: int, o: int, C: np.ndarray, F: np.ndarray, R: np.ndarray, same: np.ndarray,
                        spans: np.ndarray):
        """
        Best value, its rank sum, and back-pointers, of a segment with option ``o`` ending at ``B[k]``, per
        count layer. Of equal values, the lower rank sum wins, then the first found.
        """
        L = self.L
        best_v = np.full(L, math.inf)
        best_r = np.zeros(L, dtype=np.int64)
        best = np.full((L, 3), -1, dtype=np.int64)
        r_o = self.rank[o]
        # From the start of the record: no previous segment, a fresh period.
        if np.isfinite(C[0]) and self.tgtB[o] >= 0 and self.may_start[o]:
            t_ = self.tgtB[o]
            best_v[t_], best_r[t_], best[t_] = C[0], r_o, (0, -1, -1)
        # From a previous segment ending at B[j], j >= 1. Only a start with a finite cost can win, so when some are
        # infinite only the others are scanned, in order (one, for a fixed segmentation).
        finite = np.isfinite(C[1:k])
        js = np.arange(1, k) if finite.all() else np.flatnonzero(finite) + 1
        rows = slice(1, k) if len(js) == k - 1 else js  # a slice is a view: no copy of F when every start can win
        if len(js):
            V = F[rows] + C[rows, None, None]  # (len(js), O, L)
            V[:, ~self.allowed[:, o], :] = math.inf
            sm, sp_ = same[rows], spans[rows]
            for case, mask in (("A", sm & ~sp_), ("B", ~sm), ("C", sm & sp_)):
                if not mask.any():
                    continue
                arg, vmin, rmin = _lexargmin(V[mask], R[rows][mask])  # over j, for each (option, layer)
                jidx = js[mask][arg]
                for l in range(L):
                    if case == "A":
                        t_ = self.tgtA[o, l]
                    elif case == "B":
                        t_ = self.tgtB[o]
                    else:
                        t_ = self.tgtB[o] if self.tgtA[o, l] >= 0 else -1
                    if t_ < 0:
                        continue
                    sp = int(_lexargmin(vmin[:, l], rmin[:, l])[0])
                    v, r = vmin[sp, l], rmin[sp, l] + r_o
                    if v < best_v[t_] or (v == best_v[t_] and r < best_r[t_] and math.isfinite(v)):
                        best_v[t_], best_r[t_], best[t_] = v, r, (jidx[sp, l], sp, l)
        return best_v, best_r, best[:, 0], best[:, 1], best[:, 2]

    def refine(self, bounds: List[int], opts: List[int], window_s: float, max_sweeps: int = 5):
        """
        Move each internal boundary within ``window_s`` while that lowers the total cost (options fixed).

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
                a, b = opts[i - 1], opts[i]
                t0 = self.ts[cur]
                left = int(np.searchsorted(self.ts, t0 - window_s - 1e-9, side="left"))
                right = int(np.searchsorted(self.ts, t0 + window_s + 1e-9, side="right"))
                first, last = max(left, lo + 1), min(right, hi) - 1
                if last <= first:
                    continue

                def score(x, i=i, lo=lo, hi=hi, a=a, b=b):
                    if not (self.segment_ok(a, lo, x) and self.segment_ok(b, x, hi)):
                        return math.inf
                    if not self.feasible(bounds[:i] + [x] + bounds[i + 1:], opts):
                        return math.inf
                    return self.cost(a, lo, x) + self.cost(b, x, hi)

                seen = {cur: score(cur)}
                best_x = cur
                stride = max(1, (last - first) // 16)
                lo_x, hi_x = first, last
                while True:
                    for x in range(lo_x, hi_x + 1, stride):
                        if x not in seen:
                            seen[x] = score(x)
                    best_x = min(seen, key=seen.__getitem__)
                    if stride == 1:
                        break
                    lo_x, hi_x = max(first, best_x - stride), min(last, best_x + stride)
                    stride = max(1, stride // 8)
                if best_x != cur and seen[best_x] < seen[cur] - 1e-12 * max(1.0, abs(seen[cur])):
                    bounds[i] = best_x
                    moved.add(i)
            dirty = {j for i in moved for j in (i - 1, i + 1) if 0 < j < len(bounds) - 1}
        return bounds


def _duration_bounds(names: Sequence[str], min_duration_s: Optional[Mapping[str, float]],
                     max_duration_s: Optional[Mapping[str, float]]) -> Tuple[Tuple[float, str], Tuple[float, str]]:
    """
    The duration bounds that bind a segment of ``names`` (a state, or a
    state and its subtype; all of their bounds bind): ``((min, name), (max,
    name))``, each with the first of ``names`` it comes from.
    """
    lo = max(((float((min_duration_s or {}).get(k, 0.0)), k) for k in names), key=lambda b: b[0])
    hi = min(((float((max_duration_s or {}).get(k, math.inf)), k) for k in names), key=lambda b: b[0])
    return lo, hi


def _options(states: Sequence[str], pen: Mapping[str, float], max_counts, transitions, min_duration_s,
             max_duration_s, subtypes: Mapping[str, "Subtype"], conditions: "_Conditions", initial=None, final=None):
    """
    The program's options, counters, ``allowed`` transitions between options,
    the options the first and the last segment may take, and the result's
    count columns, from the caller's constraints. A state without subtypes is
    one option; a state with subtypes is one option per subtype, in
    declaration order, and no other (ADR 0001). A count cap is a counter over
    the options of its name: all of a state's, or a subtype's own. Every
    other constraint names states or subtypes in the same way: a state's name
    stands for all of its options. A subtype's durations and its state's
    both bind, and ``pen`` holds every state's and subtype's penalty.
    """
    options: List[_Option] = []
    options_of: Dict[str, FrozenSet[int]] = {}

    for st in states:
        mine = [(name, sub) for name, sub in subtypes.items() if sub.state.value == st]
        first = len(options)
        if not mine:
            (lo, _), (hi, _) = _duration_bounds((st,), min_duration_s, max_duration_s)
            options.append(_Option(st, pen[st], lo, hi))
        for rank, (name, sub) in enumerate(mine):
            (lo, _), (hi, _) = _duration_bounds((st, name), min_duration_s, max_duration_s)
            options.append(_Option(st, pen[name], lo, hi, accepts=conditions.of(sub), subtype=name, rank=rank))
            options_of[name] = frozenset([len(options) - 1])
        options_of[st] = frozenset(range(first, len(options)))
    names = list(states) + list(subtypes)  # states first, so that without subtypes the counters are as before
    counters = [_Counter(options_of[name], int(max_counts[name])) for name in names if name in (max_counts or {})]
    if transitions is None:
        allowed = np.ones((len(options), len(options)), dtype=bool)
    else:
        allowed = np.zeros((len(options), len(options)), dtype=bool)
        for a, nexts in transitions.items():
            for b in nexts:
                for i in options_of[a]:
                    allowed[i, list(options_of[b])] = True

    def any_of(names) -> np.ndarray:
        if names is None:
            return np.ones(len(options), dtype=bool)
        out = np.zeros(len(options), dtype=bool)
        for name in names:
            out[list(options_of[name])] = True
        return out

    return options, counters, allowed, any_of(initial), any_of(final), [(name, options_of[name]) for name in names]


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
    return np.unique(np.clip(np.asarray(pts + [0, n], dtype=np.int64), 0, n))


def _checked_record(t, y) -> Tuple[np.ndarray, Any, np.ndarray]:
    """Seconds from the first sample, the DatetimeIndex (or None) and the signal, checked."""
    ts, idx = _times(t)
    y = np.asarray(y, dtype=float)
    if ts.shape != y.shape or ts.ndim != 1:
        raise ValueError("t and y must be one-dimensional and of the same length")
    n = len(y)
    if n == 0:
        raise ValueError("empty record")
    if not np.all(np.isfinite(ts)) or (n > 1 and not np.all(np.diff(ts) > 0)):
        raise ValueError("t must be finite and strictly increasing")
    return ts, idx, y


def _checked_subtypes(subtypes: Optional[Mapping[str, Subtype]], states: Sequence[str]) -> Dict[str, Subtype]:
    """``subtypes`` as a dict, checked: each a ``Subtype`` of one of ``states``, under a name no state has."""
    if subtypes is not None and not isinstance(subtypes, Mapping):
        raise ValueError(f"subtypes must be a mapping {{name: Subtype}}, got {subtypes!r}")
    subtypes = dict(subtypes or {})
    for name, sub in subtypes.items():
        _v.name(name, "a subtype's name")
        if not isinstance(sub, Subtype):
            raise ValueError(f"subtypes[{name!r}] must be a Subtype, got {sub!r}")
        if name in _STATE_NAMES:
            raise ValueError(f"subtype {name!r} has the name of a state; give it another name")
        if sub.state.value not in states:
            raise ValueError(f"subtype {name!r} refines state {sub.state.value!r}, which has no model")
    return subtypes


def _name_check(states: Sequence[str], subtypes: Mapping[str, Subtype], *,
                models: bool = True) -> Callable[[str, str], None]:
    """
    ``check(what, name)``: every constraint names one of ``states`` (those
    with a model, unless there are no ``models``) or a subtype.
    """
    state = "a state with a model" if models else "a state"

    def check(what: str, name: str) -> None:
        if name in states or name in subtypes:
            return
        if name in _STATE_NAMES:
            raise ValueError(f"{what} names state {name!r}, which has no model")
        raise ValueError(f"{what} names {name!r}, which is not {state} or a subtype")

    return check


def _checked_sequence(transitions, initial, final, check: Callable[[str, str], None]) -> Dict[str, List[str]]:
    """Check ``transitions``, ``initial`` and ``final``; returns those of ``initial`` and ``final`` given, sorted."""
    if transitions is not None:
        for a, nexts in transitions.items():
            for st in [a, *nexts]:
                check("transitions", st)
    initial_final = {}
    for what, names in (("initial", initial), ("final", final)):
        if names is None:
            continue
        if isinstance(names, (str, bytes)) or not isinstance(names, Collection):
            raise ValueError(f"{what} must be a collection of state or subtype names, got {names!r}")
        if not names:
            raise ValueError(f"{what} is empty: name at least one state or subtype")
        for name in names:
            check(what, name)
        initial_final[what] = sorted(set(names))
    return initial_final


def optimal_segments(t, y, models: Mapping[str, Any], *, per=None, max_counts: Optional[Mapping[str, int]] = None,
                     transitions: Optional[Mapping[str, Sequence[str]]] = None,
                     min_duration_s: Optional[Mapping[str, float]] = None,
                     max_duration_s: Optional[Mapping[str, float]] = None,
                     penalty: Union[str, float, Mapping[str, float]] = "bic", method: str = "refine",
                     candidates=None, grid_s: Optional[float] = None, refine_s: Optional[float] = None,
                     period_origin=None, prune: bool = True,
                     subtypes: Optional[Mapping[str, Subtype]] = None, initial: Optional[Collection[str]] = None,
                     final: Optional[Collection[str]] = None,
                     events: Optional[Mapping[str, Any]] = None) -> OptimalSegmentation:
    """
    Best-fit segments of ``y`` under constraints (see the module docstring).

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
        ``{state or subtype: maximum segments per period}``. A state's cap
        counts all of its segments, whatever their subtype, and both caps
        bind (``{"steady": 3, "hot": 1}``). Unlisted names are not capped.
    transitions : mapping, optional
        ``{state or subtype: [states or subtypes that may follow it]}``.
        A state's name stands for all of its subtypes, and the allowed
        pairs are the union over all entries. Default: any may follow any.
    min_duration_s, max_duration_s : mapping, optional
        Bounds on segment duration per state or subtype. A subtype's and its
        state's both bind. A state whose minimum exceeds its maximum, with
        or without subtypes, is refused (``ValueError``), as is a subtype
        whose minimum or its state's exceeds its maximum or its state's.
    penalty : "bic", float, or mapping
        Added per segment (per state or subtype with a mapping; a state
        missing from it gets 0). A subtype's penalty replaces its state's;
        a subtype not in the mapping, or with ``"bic"``, takes its state's.
        See the module docstring.
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
    subtypes : mapping, optional
        ``{name: Subtype}``: named subtypes of the states in ``models`` (see
        "Subtypes" in the module docstring). Names must differ from the
        state names. Once a state has subtypes, every segment of it takes
        one of them; segments of other states have ``subtype=None``.
    initial, final : collection of str, optional
        The states or subtypes the record's first and last segments may
        take (a state's name stands for all of its subtypes). Default: any.
        A one-segment record must satisfy both.
    events : mapping, optional
        ``{name: boolean mask}``, each mask with one value per sample of
        ``t``, flagging the samples at which an event occurred (the heater
        switched on, say). A subtype's ``events`` condition names these
        masks, and every mask it names must be given; there is no unnamed
        single mask. ``info["events"]`` holds the masks used.

    Returns
    -------
    OptimalSegmentation
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    ts, idx, y = _checked_record(t, y)
    n = len(y)
    if not isinstance(models, Mapping) or not models:
        raise ValueError("models must be a non-empty mapping {state: model spec}")
    states = list(models)
    for st in states:
        if st not in _STATE_NAMES:
            raise ValueError(f"models names {st!r}, which is not a state ({', '.join(_STATE_NAMES)})")
    _fit.validate_models({"y": dict(models)}, states=states)
    subtypes = _checked_subtypes(subtypes, states)
    check = _name_check(states, subtypes)
    for name, mapping in (("max_counts", max_counts), ("min_duration_s", min_duration_s),
                          ("max_duration_s", max_duration_s)):
        for st, v in (mapping or {}).items():
            check(name, st)
            if name == "max_counts" and (isinstance(v, bool) or int(v) != v or v < 0):
                raise ValueError(f"max_counts[{st!r}] must be a non-negative integer, got {v!r}")
            if name != "max_counts" and not (float(v) >= 0):
                raise ValueError(f"{name}[{st!r}] must be >= 0, got {v!r}")
    # A state's durations, and a subtype's with its state's (both bind), must overlap.
    for what, name, names in ([("state", st, (st,)) for st in states]
                              + [("subtype", k, (sub.state.value, k)) for k, sub in subtypes.items()]):
        lo, hi = _duration_bounds(names, min_duration_s, max_duration_s)
        if lo[0] > hi[0]:
            raise ValueError(f"{what} {name!r} can take no segment: min_duration_s {lo[0]} s ({lo[1]}) > "
                             f"max_duration_s {hi[0]} s ({hi[1]})")
    initial_final = _checked_sequence(transitions, initial, final, check)
    masks = _event_masks(events, n, subtypes)
    # Penalty per state, then per subtype: a subtype's own replaces its state's, which it takes otherwise.
    own: Mapping[str, float] = {}
    if isinstance(penalty, str):
        if penalty != "bic":
            raise ValueError(f"penalty must be 'bic', a number or a mapping, got {penalty!r}")
        var = _noise_var(y)
        n_valid = int(np.isfinite(y).sum())
        pen = {st: var * (_n_params(models, st, ts, y) + 1) * math.log(max(n_valid, 2)) for st in states}
    elif isinstance(penalty, Mapping):
        for name in penalty:
            check("penalty", name)
        pen = {st: float(penalty.get(st, 0.0)) for st in states}
        own = penalty
    else:
        pen = {st: float(penalty) for st in states}
    pen.update({name: float(own.get(name, pen[sub.state.value])) for name, sub in subtypes.items()})
    if any(not (v >= 0 and math.isfinite(v)) for v in pen.values()):
        raise ValueError(f"penalties must be finite and >= 0, got {pen}")

    options, counters, allowed, may_start, may_end, columns = _options(
        states, pen, max_counts, transitions, min_duration_s, max_duration_s, subtypes, _Conditions(ts, y, masks),
        initial, final)
    prob = _Problem(ts, y, idx, models, options, counters, allowed, may_start, may_end, columns, per, period_origin,
                    max_counts, transitions, initial_final)
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

    slow = any(not prob.costs.is_fast(st) for st in states)
    # A feasible solution first, with cheap constant-fit costs; its true cost bounds the search.
    ub = prob.pruning_bound_or_raise(B) if prune and slow else math.inf
    bounds, opts = prob.solve(B, ub=ub, prune=prune)
    if method == "refine":
        if refine_s is None:
            gaps = np.diff(ts[np.minimum(B, n - 1)])
            refine_s = 0.5 * float(np.median(gaps[gaps > 0])) if (gaps > 0).any() else 0.0
        bounds = prob.refine(bounds, opts, float(refine_s))
        if subtypes:
            # A moved segment can meet an earlier-declared subtype's conditions: type the final segments again,
            # by the same program with their breakpoints and states fixed, under every constraint.
            fixed = [Segment(bounds[i], bounds[i + 1], State(options[o].state)) for i, o in enumerate(opts)]
            retype = _Problem(ts, y, idx, models, options, counters, allowed, may_start, may_end, columns, per,
                              period_origin, max_counts, transitions, initial_final, fixed_segments=fixed)
            _, opts = retype.solve(np.asarray(bounds, dtype=np.int64), prune=False)
    objective, sse = prob.total(bounds, opts)
    segs = [Segment(bounds[i], bounds[i + 1], State(options[o].state), subtype=options[o].subtype)
            for i, o in enumerate(opts)]
    n_valid = int(np.isfinite(y).sum())
    counts = prob.counts(bounds, opts)
    return OptimalSegmentation(segs, method, float(objective), float(sse),
                               math.sqrt(sse / n_valid) if n_valid else float("nan"), n_valid, pen, counts,
                               int(len(B)), prob.costs.n_fits,
                               {"upper_bound": ub, "refine_s": refine_s, "subtypes": subtypes, "events": masks},
                               subtypes)


def classify_segments(t, y, segments: Sequence[Segment], *, subtypes: Mapping[str, Subtype],
                      events: Optional[Mapping[str, Any]] = None,
                      transitions: Optional[Mapping[str, Sequence[str]]] = None,
                      initial: Optional[Collection[str]] = None,
                      final: Optional[Collection[str]] = None) -> List[Segment]:
    """
    Subtypes for segments the analyst already has (the detector's, say),
    by the same definitions as ``optimal_segments``, so that their subtypes
    and the optimiser's constraints share one definition.

    It runs the optimiser's dynamic program with the breakpoints and states
    fixed and no count caps, so a segment takes a subtype whose conditions
    it meets, the sequence obeys ``transitions``, ``initial`` and ``final``
    (a state's name stands for all of its subtypes, as there), and of the
    choices that do, the one with the least sum of the subtypes' positions
    among their states' wins. So no segment could, on its own, take an
    earlier-declared subtype whose conditions it meets without breaking
    ``transitions``, ``initial`` or ``final``; without them, each segment
    takes the first declared subtype it meets. A transient subtype with no
    conditions is told apart from another by its neighbours, as in the
    optimiser. Count caps, durations and penalties constrain the
    optimiser's choice of segments, not what a subtype means, so there are
    none here. On the optimiser's own segments, by any method, it returns
    the optimiser's subtypes when those constraints name states only; a
    cap on a subtype, say, can make the optimiser put a segment in a
    catch-all that this gives the capped subtype.

    Parameters
    ----------
    t, y
        Sample times and the signal, as for ``optimal_segments``.
    segments : sequence of Segment
        Segments covering the record in order (``states.segments``, or an
        ``OptimalSegmentation``'s). Any subtype they already carry is
        replaced.
    subtypes : mapping
        ``{name: Subtype}``, as for ``optimal_segments``. Once a state has
        subtypes, every segment of it takes one (ADR 0001); segments of
        other states get ``subtype=None``.
    events, transitions, initial, final
        As for ``optimal_segments``.

    Returns
    -------
    list of Segment
        The segments with ``subtype`` set; boundaries, states and
        ``forced`` unchanged.

    Raises
    ------
    ValueError
        When no choice of subtypes satisfies the constraints, naming, as the
        optimiser's error does, the earliest segment boundary after which
        none can continue, and the transitions, ``initial`` and ``final`` in
        force.
    """
    ts, idx, y = _checked_record(t, y)
    n = len(y)
    segs = check_segments(segments, n, require_full=True)
    for i, s in enumerate(segs):
        if s.state not in _STATE_NAMES:
            raise ValueError(f"segment {i} has state {s.state!r}, which is not a state ({', '.join(_STATE_NAMES)})")
    states = list(_STATE_NAMES)
    subs = _checked_subtypes(subtypes, states)
    initial_final = _checked_sequence(transitions, initial, final, _name_check(states, subs, models=False))
    masks = _event_masks(events, n, subs)
    no_penalty = {name: 0.0 for name in states + list(subs)}
    options, counters, allowed, may_start, may_end, columns = _options(
        states, no_penalty, max_counts=None, transitions=transitions, min_duration_s=None, max_duration_s=None,
        subtypes=subs, conditions=_Conditions(ts, y, masks), initial=initial, final=final)
    prob = _Problem(ts, y, idx, {}, options, counters, allowed, may_start, may_end, columns, per=None,
                    period_origin=None, transitions=transitions, initial_final=initial_final, fixed_segments=segs)
    _, opts = prob.solve(np.array([0] + [s.stop for s in segs], dtype=np.int64), prune=False)
    return [replace(s, subtype=options[o].subtype) for s, o in zip(segs, opts)]


__all__ = ["optimal_segments", "OptimalSegmentation", "Subtype", "classify_segments", "compare_segments",
           "count_by_period", "METHODS"]

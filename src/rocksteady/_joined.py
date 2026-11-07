"""
Joined transients: the costs and the dynamic program of ``optimal_segments(..., join_transients=True)``.

A joined transient is the straight line from the level of the steady segment before it, at its first sample,
to the level of the segment after it, at that segment's first sample. At the record's start or end only its
inner side is joined, and its slope is fitted by least squares through that point; a transient that is the
whole record is a free line. A steady segment is fitted as without the option: its level is the mean of its
finite samples (ADR 0002).

The SSE of a transient over ``[p, q)`` joined to the levels ``L1`` and ``L2`` splits into three terms, none
of which can cancel another:

    F + A (L2 - L1 - s)^2 + n (ybar - (1 - ubar) L1 - ubar L2)^2

``F`` is the SSE of the segment's free least-squares line, ``A (L2 - L1 - s)^2`` what the joined line's rise
costs against the free line's rise ``s``, and the last term what the joined line's mean costs against the
samples' mean ``ybar``; ``ubar`` is where the samples' mean time lies between the two joined times, as a
fraction of the time between them. They come from the range sums of ``optimal._Costs``, and a segment whose
time variance the sums do not resolve is summed directly.

A transient's cost depends on the levels of both its neighbours, so the program's state holds the last
segment's start as well as its end: a steady segment ``[B[s], B[k])`` is one state, its level known. The
program runs over ``s``. At ``s`` every state ending there is final, and it costs every steady segment
starting there, for every end at once: after the previous steady segment directly, after a transient that
starts the record, or after a transient that follows a steady segment ending at some earlier ``B[j]``. The
last needs the minimum, over those earlier steady segments, of a cost that couples their level to the new
one; it is taken over the candidates' levels all at once.
"""

import math
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Sequence, Tuple, cast

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .optimal import _Costs, _Problem

_EPS = float(np.finfo(float).eps)
_CHUNK = 1 << 21  # elements of one block of the coupled minimum


class Terms(NamedTuple):
    """The terms of a transient's joined SSE over ``[p, q)``, one per start (see the module docstring)."""

    n: np.ndarray
    F: np.ndarray
    A: np.ndarray
    s: np.ndarray
    ybar: np.ndarray
    ubar: np.ndarray


def terms(costs: "_Costs", key: str, ps: np.ndarray, q: int, pin: int) -> Terms:
    """
    The terms of each transient ``[p, q)`` for the starts ``ps``, its far end pinned at the time of sample
    ``pin`` (``q`` within the record; any later sample for one that ends the record, whose slope is free).
    ``key`` names a linear model in ``costs``, which gives the free line's SSE.
    """
    ps = np.asarray(ps, dtype=np.int64)
    n = (costs.N[q] - costs.N[ps]).astype(float)
    nn = np.where(n > 0, n, 1.0)
    sx, e_x = costs.Sx.range(ps, q)
    sy, _ = costs.Sy.range(ps, q)
    sxx, e_xx = costs.Sxx.range(ps, q)
    sxy, _ = costs.Sxy.range(ps, q)
    xbar, ybar = sx / nn, sy / nn
    sxx_c = sxx - sx * sx / nn
    sxy_c = sxy - sx * sy / nn
    e_sxx = e_xx + 2.0 * np.abs(sx) * e_x / nn + 4.0 * _EPS * np.abs(sxx)
    several = n > 1
    sxx_c = np.where(several, sxx_c, 0.0)
    sxy_c = np.where(several, sxy_c, 0.0)
    F = np.zeros(len(ps))
    three = np.flatnonzero(n >= 3)
    if len(three):
        F[three] = costs.vector(key, ps[three], q)  # the free line, fitted directly where the sums cannot tell
    # With two samples or more, the time variance must be resolved above its rounding; else sum directly.
    for i in np.flatnonzero(several & ~(sxx_c > 1e6 * e_sxx)):
        p = int(ps[i])
        ok = costs.ok[p:q]
        xx, yy = costs.x_all[p:q][ok], costs.yc[p:q][ok]
        xbar[i], ybar[i] = xx.mean(), yy.mean()
        dx, dy = xx - xbar[i], yy - ybar[i]
        sxx_c[i], sxy_c[i] = float(dx @ dx), float(dx @ dy)
        if n[i] >= 3:
            r = dy - (sxy_c[i] / sxx_c[i]) * dx if sxx_c[i] > 0 else dy
            F[i] = float(r @ r)
    xp = costs.x_all[ps]
    D = costs.x_all[pin] - xp
    D = np.where(D > 0, D, 1.0)
    has_var = sxx_c > 0
    A = sxx_c / (D * D)
    s = np.where(has_var, D * sxy_c / np.where(has_var, sxx_c, 1.0), 0.0)
    ubar = np.where(n > 0, (xbar - xp) / D, 0.0)
    return Terms(n, F, A, s, np.where(n > 0, ybar, 0.0), ubar)


def _one_side(A, n, c, g, s):
    """``min over d of A (d - s)^2 + n (c d - g)^2``: a line pinned at one end, its slope free."""
    A, n, c, g, s = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (A, n, c, g, s)))
    den = A + n * c * c
    with np.errstate(invalid="ignore", divide="ignore"):
        sloped = A * n * (c * s - g) ** 2 / np.where(den > 0, den, 1.0)
    # Without a time variance (one sample, or none), the free slope meets the sample unless it lies at the pin.
    flat = np.where(c != 0, 0.0, n * g * g)
    return np.where(A > 0, sloped, flat)


def interior(t: Terms, L1, L2):
    """The SSE of transients joined to ``L1`` before and ``L2`` after (arrays broadcast)."""
    return t.F + t.A * (L2 - L1 - t.s) ** 2 + t.n * (t.ybar - (1.0 - t.ubar) * L1 - t.ubar * L2) ** 2


def at_start(t: Terms, L2):
    """The SSE of transients that start the record, joined to ``L2`` after."""
    return t.F + _one_side(t.A, t.n, 1.0 - t.ubar, L2 - t.ybar, t.s)


def at_end(t: Terms, L1):
    """The SSE of transients that end the record, joined to ``L1`` before."""
    return t.F + _one_side(t.A, t.n, t.ubar, t.ybar - L1, t.s)


def _nan_inf(v):
    return np.where(np.isnan(v), math.inf, v)


class Joined:
    """
    The joined fit of one problem: ``prob``'s options, counts and costs, with steady segments fitted by the
    model ``steady_key`` and transients by ``transient_key`` (a constant and a line), and ``extra``, the
    penalty for each free parameter of a transient at the record's start or end (``"bic"``), else 0.
    """

    def __init__(self, prob: "_Problem", steady_key: Optional[str], transient_key: Optional[str], extra: float):
        self.prob = prob
        self.costs = cast("_Costs", prob.costs)  # the optimiser's own costs: range sums, not a fixed segmentation
        self.skey, self.tkey = steady_key, transient_key
        self.extra = extra

    # -- costs of given segments -------------------------------------------------------------------------

    def level(self, p: int, q: int) -> float:
        """The fitted level of a steady segment ``[p, q)``, in the costs' centred units (NaN without a sample)."""
        c = self.costs
        n = c.N[q] - c.N[p]
        if n <= 0:
            return math.nan
        return float(c.Sy.range(np.array([p]), q)[0][0] / n)

    def segment_cost(self, bounds: Sequence[int], opts: Sequence[int], i: int) -> float:
        """The SSE of segment ``i`` of a segmentation, joined to its neighbours; with its penalty when asked."""
        prob = self.prob
        p, q = bounds[i], bounds[i + 1]
        o = opts[i]
        if prob.options[o].state != "transient":
            return prob.cost(o, p, q)
        before = i > 0
        after = i < len(opts) - 1
        levels: List[Optional[float]] = []
        for j, near in ((i - 1, before), (i + 1, after)):
            if not near:
                levels.append(None)
                continue
            if prob.options[opts[j]].state != "steady":
                return math.inf
            lv = self.level(bounds[j], bounds[j + 1])
            if math.isnan(lv):
                return math.inf
            levels.append(lv)
        L1, L2 = levels
        if L1 is None and L2 is None:
            return prob.cost(o, p, q)  # the whole record: a free line
        key = cast(str, self.tkey)
        if L2 is None:
            t = terms(self.costs, key, np.array([p]), q, prob.n - 1)
            return float(at_end(t, L1)[0])
        t = terms(self.costs, key, np.array([p]), q, q)
        if L1 is None:
            return float(at_start(t, L2)[0])
        return float(interior(t, L1, L2)[0])

    def free_params(self, opts: Sequence[int], i: int) -> int:
        """The free parameters a transient's joined line keeps: 0 joined on both sides, 1 on one, 2 on none."""
        return int(i == 0) + int(i == len(opts) - 1)

    def penalty(self, opts: Sequence[int], i: int) -> float:
        o = opts[i]
        pen = float(self.prob.pen[o])
        if self.prob.options[o].state == "transient":
            pen += self.extra * self.free_params(opts, i)
        return pen

    def total(self, bounds: Sequence[int], opts: Sequence[int]) -> Tuple[float, float]:
        """``(objective, sse)`` of a segmentation under the joined fit."""
        sse = sum(self.segment_cost(bounds, opts, i) for i in range(len(opts)))
        pen = sum(self.penalty(opts, i) for i in range(len(opts)))
        return sse + pen + self.prob.overflow_cost(bounds, opts), sse

    def window_cost(self, bounds: Sequence[int], opts: Sequence[int], i: int) -> float:
        """
        What moving boundary ``i`` can change: the SSE of the segments on either side of it and of the
        transients next to them, joined to their levels.
        """
        lo, hi = max(0, i - 2), min(len(opts) - 1, i + 1)
        return sum(self.segment_cost(bounds, opts, k) for k in range(lo, hi + 1))

    # -- the program ---------------------------------------------------------------------------------------

    def solve(self, B: np.ndarray, ub: float = math.inf) -> Tuple[List[int], List[int]]:
        return _Program(self, np.asarray(B, dtype=np.int64), ub).run()


def _lexmin(v: np.ndarray, r: np.ndarray):
    """Along axis 0: the index of the least ``v``, of those the least ``r``, then the first; and ``v``, ``r`` there."""
    vmin = v.min(axis=0)
    rr = np.broadcast_to(r, v.shape)
    arg = np.where(v == vmin, rr, np.iinfo(np.int64).max).argmin(axis=0)
    return arg, vmin, np.take_along_axis(rr, np.expand_dims(arg, 0), axis=0)[0]


class _Entries:
    """The program's final states: steady segments ``[B[s], B[k])``, each with its option and count layer."""

    FIELDS = ("k", "s", "o", "l", "v", "r", "L", "kind", "pred", "ot")

    def __init__(self):
        self.parts: Dict[str, List[np.ndarray]] = {f: [] for f in self.FIELDS}
        self.arrays: Optional[Dict[str, np.ndarray]] = None
        self.size = 0

    def add(self, **cols):
        size = len(cols["v"])
        for f in self.FIELDS:
            self.parts[f].append(np.asarray(cols[f]))
        self.size += size
        self.arrays = None

    def get(self) -> Dict[str, np.ndarray]:
        if self.arrays is None:
            self.arrays = {f: (np.concatenate(self.parts[f]) if self.parts[f] else np.zeros(0)) for f in self.FIELDS}
            for f in self.FIELDS:  # one part from here on
                self.parts[f] = [self.arrays[f]]
        return self.arrays


START, AFTER_STEADY, AFTER_FIRST_TRANSIENT, AFTER_TRANSIENT = 0, 1, 2, 3


class _Program:
    def __init__(self, joined: Joined, B: np.ndarray, ub: float):
        self.j = joined
        prob = self.prob = joined.prob
        self.B, self.m = B, len(B)
        self.ub = ub * (1 + 1e-9) + 1e-12 if math.isfinite(ub) else math.inf
        self.n = prob.n
        opts = prob.options
        self.steady = [o for o in range(prob.O) if opts[o].state == "steady" and prob.tgtB[o] >= 0]
        self.trans = [o for o in range(prob.O) if opts[o].state == "transient" and prob.tgtB[o] >= 0]
        self._tables()

    # The candidate segments, their validity and their costs, as (start index, end index) tables over B.
    def _tables(self):
        from .optimal import _accept_all  # pylint: disable=import-outside-toplevel

        prob, B, m = self.prob, self.B, self.m
        ts, n = prob.ts, self.n
        costs = self.j.costs
        self.V: Dict[int, np.ndarray] = {}
        for o in self.steady + self.trans:
            V = np.zeros((m, m), dtype=bool)
            opt = prob.options[o]
            for k in range(1, m):
                q = int(B[k])
                starts = B[:k]
                dur = ts[q - 1] - ts[starts] + prob.dt
                ok = (dur >= prob.min_d[o] - 1e-9) & (dur <= prob.max_d[o] + 1e-9)
                if ok.any() and opt.accepts is not _accept_all:
                    ok &= opt.accepts(starts, q)
                V[:k, k] = ok
            self.V[o] = V
        self.SS = np.full((m, m), math.inf)
        self.LV = np.full((m, m), math.nan)
        names = ("n", "F", "A", "s", "ybar", "ubar")
        self.T = {f: np.zeros((m, m)) for f in names}
        self.E = {f: np.zeros(m) for f in names}  # transients that end the record, by start
        any_steady = np.zeros((m, m), dtype=bool)
        for o in self.steady:
            any_steady |= self.V[o]
        any_trans = np.zeros((m, m), dtype=bool)
        for o in self.trans:
            any_trans |= self.V[o]
        for k in range(1, m):
            q = int(B[k])
            starts = B[:k]
            if self.steady and any_steady[:k, k].any():
                self.SS[:k, k] = costs.vector(cast(str, self.j.skey), starts, q)
                cnt = costs.N[q] - costs.N[starts]
                sy, _ = costs.Sy.range(starts, q)
                with np.errstate(invalid="ignore", divide="ignore"):
                    self.LV[:k, k] = np.where(cnt > 0, sy / np.where(cnt > 0, cnt, 1.0), math.nan)
            if self.trans and any_trans[:k, k].any():
                if q < n:
                    t = terms(costs, cast(str, self.j.tkey), starts, q, q)
                    for f in names:
                        self.T[f][:k, k] = getattr(t, f)
                else:
                    t = terms(costs, cast(str, self.j.tkey), starts, q, n - 1)
                    for f in names:
                        self.E[f][:k] = getattr(t, f)
                    self.T["F"][:k, k] = t.F  # for the bound on transients
        self.SS = np.where(self.SS <= self.ub, self.SS, math.inf)

    def step(self, o: int, P, Q, lprev):
        """
        The count layer after a segment of option ``o`` over ``[P, Q)`` that follows a segment ending at
        ``P``, in layer ``lprev`` (arrays broadcast), and the overflow cost it adds: -1 where a cap forbids it.
        """
        prob = self.prob
        pid = prob.pid
        P, Q, lprev = np.broadcast_arrays(np.asarray(P, dtype=np.int64), np.asarray(Q, dtype=np.int64),
                                          np.asarray(lprev, dtype=np.int64))
        same = pid[P] == pid[np.maximum(P - 1, 0)]
        spans = pid[Q - 1] > pid[P]
        tA = prob.tgtA[o, lprev]
        tB = prob.tgtB[o]
        tgt = np.where(~same, tB, np.where(spans, np.where(tA >= 0, tB, -1), tA))
        over = np.where(same, prob.in_progress[o, lprev], 0.0) + prob.per_period[o] * (pid[Q - 1] - pid[P] + 1)
        return tgt, over

    def first(self, o: int, Q):
        """The count layer and overflow of a segment of option ``o`` that starts the record, ending at ``Q``."""
        prob = self.prob
        Q = np.asarray(Q, dtype=np.int64)
        tgt = np.full(Q.shape, prob.tgtB[o])
        return tgt, prob.per_period[o] * (prob.pid[Q - 1] - prob.pid[0] + 1)

    def run(self) -> Tuple[List[int], List[int]]:
        prob, B, m, L = self.prob, self.B, self.m, self.prob.L
        entries = _Entries()
        pending: List[List[Dict[str, np.ndarray]]] = [[] for _ in range(m)]  # states ending at each B[k]
        final: Optional[Tuple[float, int, Tuple]] = None  # (value, rank, description)
        last_reached = 0

        def offer(v, r, desc):
            nonlocal final
            if not math.isfinite(v):
                return
            if final is None or v < final[0] or (v == final[0] and r < final[1]):
                final = (float(v), int(r), desc)

        for s in range(m - 1):
            P = int(B[s])
            if s >= 1:
                for part in pending[s]:
                    entries.add(**part)
                pending[s] = []
            E = entries.get()
            mine = np.flatnonzero(E["k"] == s) if s >= 1 else np.zeros(0, dtype=np.int64)
            if len(mine):
                last_reached = s
            # The best state ending at B[s] for each option and layer: what a steady segment after it adds to.
            best_at: List[Tuple[int, int, float, int, int]] = []
            if len(mine):
                key = E["o"][mine] * L + E["l"][mine]
                order = np.lexsort((mine, E["r"][mine], E["v"][mine], key))
                sk = key[order]
                firsts = order[np.concatenate(([True], sk[1:] != sk[:-1]))]
                for i in mine[firsts]:
                    best_at.append((int(E["o"][i]), int(E["l"][i]), float(E["v"][i]), int(E["r"][i]), int(i)))
                self._ends(E, mine, P, offer)
            groups, arrived = self._transient_groups(E, s, P) if s >= 1 else ([], False)
            if arrived:  # a transient can end here, whether or not a steady segment can follow it
                last_reached = s
            ks = np.arange(s + 1, m)
            Q = B[ks]
            L2 = self.LV[s, ks]
            for o in self.steady:
                valid = self.V[o][s, ks] & ((ks < m - 1) | prob.may_end[o]) & np.isfinite(self.SS[s, ks])
                if not valid.any():
                    continue
                base = np.where(valid, self.SS[s, ks] + prob.pen[o], math.inf)
                K = len(ks)
                bv = np.full((K, L), math.inf)
                br = np.zeros((K, L), dtype=np.int64)
                bkind = np.full((K, L), -1, dtype=np.int64)
                bpred = np.full((K, L), -1, dtype=np.int64)
                bot = np.full((K, L), -1, dtype=np.int64)
                rows = np.arange(K)

                def put(v, r, tgt, kind, pred, ot):
                    v = np.where((tgt >= 0) & valid, v, math.inf)
                    t_ = np.maximum(tgt, 0)
                    cur, cr = bv[rows, t_], br[rows, t_]
                    r = np.broadcast_to(r, v.shape)
                    win = np.isfinite(v) & ((v < cur) | ((v == cur) & (r < cr)))
                    idx = np.flatnonzero(win)
                    if not len(idx):
                        return
                    bv[idx, t_[idx]] = v[idx]
                    br[idx, t_[idx]] = r[idx]
                    bkind[idx, t_[idx]] = kind
                    bpred[idx, t_[idx]] = np.broadcast_to(pred, v.shape)[idx]
                    bot[idx, t_[idx]] = ot

                rank_o = int(prob.rank[o])
                if s == 0 and prob.may_start[o]:
                    tgt, over = self.first(o, Q)
                    put(base + over, rank_o, tgt, START, -1, -1)
                for po, pl, pv, pr, pi in best_at:
                    if not prob.allowed[po, o]:
                        continue
                    tgt, over = self.step(o, P, Q, pl)
                    put(pv + base + over, pr + rank_o, tgt, AFTER_STEADY, pi, -1)
                for ot, lt, kind, gv, gr, gp in groups:
                    if not prob.allowed[ot, o]:
                        continue
                    tgt, over = self.step(o, P, Q, lt)
                    put(gv + base + over, gr + rank_o, tgt, kind, gp, ot)
                fin = np.isfinite(bv) & (bv <= self.ub)
                kk, ll = np.nonzero(fin)
                if not len(kk):
                    continue
                for k in np.unique(kk):
                    sel = kk == k
                    kv, lv = kk[sel], ll[sel]
                    pending[int(ks[k])].append({
                        "k": np.full(len(kv), int(ks[k])), "s": np.full(len(kv), s), "o": np.full(len(kv), o),
                        "l": lv, "v": bv[kv, lv], "r": br[kv, lv], "L": np.full(len(kv), L2[k]),
                        "kind": bkind[kv, lv], "pred": bpred[kv, lv], "ot": bot[kv, lv]})
        for part in pending[m - 1]:
            entries.add(**part)
        E = entries.get()
        mine = np.flatnonzero(E["k"] == m - 1)
        for i in mine:
            offer(E["v"][i], E["r"][i], ("steady", int(i)))
        if len(mine):
            last_reached = m - 1
        for ot in self.trans:  # a transient that is the whole record: a free line
            if not (prob.may_start[ot] and prob.may_end[ot] and self.V[ot][0, m - 1]):
                continue
            tgt, over = self.first(ot, np.array([self.n]))
            if tgt[0] < 0:
                continue
            c = prob.cost(ot, 0, self.n)
            offer(c + prob.pen[ot] + 2 * self.j.extra + float(over[0]), int(prob.rank[ot]), ("whole", ot))
        if final is None:
            raise ValueError(prob._infeasible(B, last_reached) + " With join_transients, a transient cannot border "
                             "a steady segment that has no finite sample.")
        return self._backtrack(E, final[2])

    def _ends(self, E, mine, P: int, offer):
        """Offer each transient ``[P, n)`` after the states ending at ``P``, joined to their levels."""
        prob, m = self.prob, self.m
        if P >= self.n:
            return
        s = int(np.searchsorted(self.B, P))
        t = Terms(*(np.full(len(mine), self.E[f][s]) for f in Terms._fields))
        L1 = E["L"][mine]
        for ot in self.trans:
            if not (prob.may_end[ot] and self.V[ot][s, m - 1]):
                continue
            ok = prob.allowed[E["o"][mine].astype(np.int64), ot] & np.isfinite(L1)
            if not ok.any():
                continue
            tgt, over = self.step(ot, P, self.n, E["l"][mine])
            v = E["v"][mine] + prob.pen[ot] + self.j.extra + over + _nan_inf(at_end(t, L1))
            v = np.where(ok & (tgt >= 0), v, math.inf)
            r = E["r"][mine] + prob.rank[ot]
            for i in np.flatnonzero(np.isfinite(v)):
                offer(v[i], r[i], ("end", int(mine[i]), ot))

    def _transient_groups(self, E, s: int, P: int):
        """
        For the steady segments starting at ``B[s]``, the best way to arrive by a transient ending there, for
        each end: ``(ot, layer, kind, value (K,), rank (K,), state (K,))``, one per transient option and count
        layer after it. The values include the transient's cost joined to each end's level. Also whether any
        transient can end at ``B[s]`` at all.
        """
        prob, B, m = self.prob, self.B, self.m
        ks = np.arange(s + 1, m)
        L2 = self.LV[s, ks]
        out = []
        arrived = False
        T = self.T
        # A transient that starts the record, then this steady segment.
        t0 = Terms(*(np.full(len(ks), T[f][0, s]) for f in Terms._fields))
        for ot in self.trans:
            if not (prob.may_start[ot] and self.V[ot][0, s]):
                continue
            tgt, over = self.first(ot, np.array([P]))
            if tgt[0] < 0:
                continue
            arrived = True
            v = _nan_inf(at_start(t0, L2)) + prob.pen[ot] + self.j.extra + float(over[0])
            out.append((ot, int(tgt[0]), AFTER_FIRST_TRANSIENT, v, np.full(len(ks), int(prob.rank[ot])),
                        np.full(len(ks), -1)))
        # A transient [B[j], B[s]) after a steady segment ending at B[j].
        before = np.flatnonzero((E["k"] >= 1) & (E["k"] < s) & np.isfinite(E["L"]))
        if not len(before):
            return out, arrived
        jk = E["k"][before].astype(np.int64)
        for ot in self.trans:
            ok = self.V[ot][jk, s] & prob.allowed[E["o"][before].astype(np.int64), ot]
            if not ok.any():
                continue
            e = before[ok]
            j = jk[ok]
            tgt, over = self.step(ot, B[j], P, E["l"][e])
            base = E["v"][e] + prob.pen[ot] + over + T["F"][j, s]
            keep = (tgt >= 0) & (base <= self.ub)
            if not keep.any():
                continue
            arrived = True
            e, j, tgt, base = e[keep], j[keep], tgt[keep], base[keep]
            L1 = E["L"][e]
            A, n, ub_ = T["A"][j, s], T["n"][j, s], T["ubar"][j, s]
            a1 = L1 + T["s"][j, s]
            b0 = T["ybar"][j, s] - (1.0 - ub_) * L1
            rank = E["r"][e] + prob.rank[ot]
            for lt in np.unique(tgt):
                g = np.flatnonzero(tgt == lt)
                bestv = np.full(len(ks), math.inf)
                bestr = np.zeros(len(ks), dtype=np.int64)
                bestp = np.full(len(ks), -1, dtype=np.int64)
                step_rows = max(1, _CHUNK // max(1, len(ks)))
                for c0 in range(0, len(g), step_rows):
                    gg = g[c0:c0 + step_rows]
                    vals = (base[gg, None] + A[gg, None] * (L2[None, :] - a1[gg, None]) ** 2
                            + n[gg, None] * (b0[gg, None] - ub_[gg, None] * L2[None, :]) ** 2)
                    vals = _nan_inf(vals)
                    arg, vmin, rmin = _lexmin(vals, rank[gg, None])
                    win = (vmin < bestv) | ((vmin == bestv) & (rmin < bestr))
                    bestv = np.where(win, vmin, bestv)
                    bestr = np.where(win, rmin, bestr)
                    bestp = np.where(win, e[gg][arg], bestp)
                out.append((ot, int(lt), AFTER_TRANSIENT, bestv, bestr, bestp))
        return out, arrived

    def _backtrack(self, E, desc) -> Tuple[List[int], List[int]]:
        B = self.B
        segs: List[Tuple[int, int, int]] = []  # (start, stop, option), last first

        def walk(i: int):
            while True:
                k, s, o = int(E["k"][i]), int(E["s"][i]), int(E["o"][i])
                segs.append((int(B[s]), int(B[k]), o))
                kind, pred, ot = int(E["kind"][i]), int(E["pred"][i]), int(E["ot"][i])
                if kind == START:
                    return
                if kind == AFTER_STEADY:
                    i = pred
                    continue
                if kind == AFTER_FIRST_TRANSIENT:
                    segs.append((0, int(B[s]), ot))
                    return
                segs.append((int(B[int(E["k"][pred])]), int(B[s]), ot))
                i = pred

        if desc[0] == "whole":
            return [0, self.n], [desc[1]]
        if desc[0] == "end":
            _, i, ot = desc
            segs.append((int(B[int(E["k"][i])]), self.n, ot))
            walk(i)
        else:
            walk(desc[1])
        segs.reverse()
        return [a for a, _, _ in segs] + [segs[-1][1]], [o for _, _, o in segs]

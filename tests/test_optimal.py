"""Constrained optimal segmentation: best-fit segments under constraints (#45)."""

import dataclasses
import itertools
import math
import pickle
import re

import numpy as np
import pandas as pd
import pytest

import rocksteady as rs
from rocksteady.fit import fit_model, fit_poly
from rocksteady.optimal import compare_segments, count_by_period, optimal_segments
from rocksteady.states import Segment, State

ST, TR = State.STEADY, State.TRANSIENT


# -- an independent reference: brute force over every segmentation -----------------


def _sse(spec, t, y, state):
    r = fit_model(spec, t, y, state)
    e = r.residuals[np.isfinite(r.residuals)]
    if not np.isfinite(y).any():
        return 0.0
    return float(e @ e) if len(e) else math.inf


def _brute(t, y, models, per=None, max_counts=None, transitions=None, min_d=None, penalty=0.0, positions=None,
           subtypes=None, events=None):
    """
    The best objective over every segmentation and every choice for each segment, by exhaustive depth-first
    search: a choice is a state without subtypes, or one subtype of a state that has some. It cuts a branch
    only when it breaks a constraint, never on cost. Returns (objective, (bounds, choices)).
    """
    n = len(y)
    dt = float(np.median(np.diff(t)))
    pid = np.zeros(n, dtype=int) if per is None else np.floor(t / per).astype(int)
    cuts = set(range(1, n + 1)) if positions is None else {p for p in positions if 0 < p <= n} | {n}
    subtypes = subtypes or {}
    choices = []  # (name, state, subtype): name is the subtype, or the state when it has none
    for st in models:
        mine = [(k, s) for k, s in subtypes.items() if s.state.value == st]
        choices += [(k, st, s) for k, s in mine] or [(st, st, None)]
    caps = dict(max_counts or {})
    sse = {}

    def cost(st, p, q):
        if (st, p, q) not in sse:
            sse[(st, p, q)] = _sse(models[st], t[p:q], y[p:q], st)
        return sse[(st, p, q)]

    def meets(sub, p, q):
        return sub is None or _meets_conditions(sub, t[p:q], y[p:q], _within(events, slice(p, q)))

    best = [math.inf, None]

    def search(p, prev, counts, acc, path):
        if p == n:
            if acc < best[0] - 1e-9:
                best[:] = [acc, path]
            return
        for q in sorted(c for c in cuts if c > p):
            for name, st, sub in choices:
                if min_d and t[q - 1] - t[p] + dt < min_d.get(st, 0) - 1e-9:
                    continue
                if transitions is not None and prev is not None and st not in transitions[prev]:
                    continue
                if not meets(sub, p, q):
                    continue
                new = dict(counts)
                for key in {st, name}:
                    if key in caps:
                        for pp in range(pid[p], pid[q - 1] + 1):
                            new[(pp, key)] = new.get((pp, key), 0) + 1
                if any(new[(pp, key)] > caps[key] for pp, key in new):
                    continue
                search(q, st, new, acc + cost(st, p, q) + penalty, path + [(p, q, name)])

    search(0, None, {}, 0.0, [])
    if best[1] is None:
        return math.inf, None
    path = best[1]
    return best[0], ([0] + [q for _, q, _ in path], tuple(name for _, _, name in path))


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("case", ["plain", "periods", "transitions", "min_duration"])
def test_exact_matches_brute_force(seed, case):
    rng = np.random.default_rng(seed)
    n = 8
    t = np.arange(n, dtype=float)
    y = np.where(t < rng.integers(2, 6), 0.0, 3.0) + rng.normal(0, 0.3, n)
    models = {"steady": 0, "transient": 1}
    kw = dict(penalty=0.5)
    if case == "periods":
        kw.update(per=3.0, max_counts={"steady": 1, "transient": 1})
    elif case == "transitions":
        kw.update(transitions={"steady": ["transient"], "transient": ["steady"]}, max_counts={"steady": 2})
    elif case == "min_duration":
        kw.update(min_duration_s={"steady": 3.0, "transient": 2.0})
    want_cost, (want_b, want_s) = _brute(t, y, models, per=kw.get("per"), max_counts=kw.get("max_counts"),
                                         transitions=kw.get("transitions"), min_d=kw.get("min_duration_s"),
                                         penalty=0.5)
    got = optimal_segments(t, y, models, method="exact", **kw)
    assert got.objective == pytest.approx(want_cost, rel=1e-9, abs=1e-9)


def test_exact_matches_brute_force_with_a_nonlinear_model():
    t = np.arange(14.0)
    y = np.where(t < 4, 0.0, 5.0 * (1 - np.exp(-(t - 4) / 2.0))) + np.random.default_rng(9).normal(0, 0.05, 14)
    models = {"steady": 0, "transient": {"model": "first_order", "tau_min": 0.5, "tau_max": 50.0}}
    positions = [0, 2, 4, 6, 8, 11, 14]
    want_cost, _ = _brute(t, y, models, max_counts={"steady": 2, "transient": 1}, penalty=0.1, positions=positions)
    for prune in (True, False):
        got = optimal_segments(t, y, models, method="subset", candidates=positions, penalty=0.1,
                               max_counts={"steady": 2, "transient": 1}, prune=prune)
        assert got.objective == pytest.approx(want_cost, rel=1e-9)


def test_fast_costs_equal_the_fitted_ones():
    rng = np.random.default_rng(1)
    t = 1.7e9 + np.arange(300.0)  # epoch-sized times
    y = 400 + 0.05 * (t - t[0]) + rng.normal(0, 1, 300)
    y[[10, 11, 150]] = np.nan
    from rocksteady.optimal import _Costs

    for order in (0, 1):
        c = _Costs(t - t[0], y, {"steady": order}, ["steady"])
        for p, q in [(0, 300), (5, 12), (148, 152), (290, 300), (10, 12), (100, 101)]:
            r = fit_poly(t[p:q], y[p:q], order)
            e = r.residuals[np.isfinite(r.residuals)]
            assert c.cost("steady", p, q) == pytest.approx(float(e @ e), rel=1e-7, abs=1e-6)


def test_ramp_optima_are_not_nested():
    """Why "subset" is an approximation: the best 2- and 3-segment splits share no breakpoint."""
    n = 120
    t = np.arange(n, dtype=float)
    y = t / n
    two = optimal_segments(t, y, {"steady": 0}, method="exact", max_counts={"steady": 2}, penalty=0)
    three = optimal_segments(t, y, {"steady": 0}, method="exact", max_counts={"steady": 3}, penalty=0)
    assert [s.start for s in two.segments][1:] == [60]
    assert [s.start for s in three.segments][1:] == [40, 80]
    assert not {60} & {40, 80}
    # Restricted to the 3-segment breakpoints, the best 2-segment split is worse than the true optimum.
    sub = optimal_segments(t, y, {"steady": 0}, method="subset", candidates=[40, 80], max_counts={"steady": 2},
                           penalty=0)
    assert sub.objective > two.objective * 1.1


# -- counting -------------------------------------------------------------------------


def test_spanning_segments_count_against_every_period_they_touch():
    idx = pd.date_range("2026-03-02 00:00", "2026-03-05 23:50", freq="10min", tz="America/Chicago")
    pos = {ts: i for i, ts in enumerate(idx)}

    def at(s):
        return pos[pd.Timestamp(s, tz="America/Chicago")]

    segs = [Segment(0, at("2026-03-02 08:00"), ST),                          # zero, Monday morning
            Segment(at("2026-03-02 08:00"), at("2026-03-02 08:30"), TR),       # startup
            Segment(at("2026-03-02 08:30"), at("2026-03-02 16:00"), ST),       # at power
            Segment(at("2026-03-02 16:00"), at("2026-03-02 16:20"), TR),       # shutdown
            Segment(at("2026-03-02 16:20"), at("2026-03-05 08:00"), ST),       # zero until Thursday
            Segment(at("2026-03-05 08:00"), len(idx), TR)]
    counts = count_by_period(segs, idx, "1D")
    assert counts["steady"].tolist() == [3, 1, 1, 1]  # Monday: zero, at power, zero (shared with Tue, Wed, Thu)
    assert counts["transient"].tolist() == [2, 0, 0, 1]
    assert counts.index[0] == pd.Timestamp("2026-03-02")  # local calendar days
    # Half-open: a segment whose last sample is 23:50 does not touch the next day.
    tue_end = at("2026-03-03 23:50") + 1
    split = [Segment(0, tue_end, ST), Segment(tue_end, len(idx), ST)]
    assert count_by_period(split, idx, "1D")["steady"].tolist() == [1, 1, 1, 1]


def test_per_period_limits_bind():
    """Two plateaus per 100 s period; at most one steady segment per period."""
    rng = np.random.default_rng(2)
    t = np.arange(400.0)
    y = np.where((t % 100) < 50, 0.0, 4.0) + rng.normal(0, 0.05, 400)
    pen = {"steady": 1.0, "transient": 20.0}  # a line always fits a plateau at least as well as a constant
    free = optimal_segments(t, y, {"steady": 0, "transient": 1}, method="exact", grid_s=10, penalty=pen)
    lim = optimal_segments(t, y, {"steady": 0, "transient": 1}, method="exact", grid_s=10, penalty=pen,
                           per=100.0, max_counts={"steady": 1})
    assert (lim.counts["steady"] <= 1).all() and (free.counts["steady"] > 1).any()
    assert lim.objective >= free.objective
    # A long steady segment counts in every period it touches.
    lim2 = optimal_segments(t, np.zeros(400), {"steady": 0}, method="exact", grid_s=50, penalty=0,
                            per=100.0, max_counts={"steady": 2})
    assert (lim2.counts["steady"] <= 2).all()


def test_maximum_counts_are_maximums_with_the_default_penalty():
    rng = np.random.default_rng(3)
    t = np.arange(300.0)
    y = np.where(t < 150, 0.0, 5.0) + rng.normal(0, 0.1, 300)
    kw = dict(method="exact", grid_s=5, max_counts={"steady": 3, "transient": 2})
    bic = optimal_segments(t, y, {"steady": 0, "transient": 1}, **kw)
    filled = optimal_segments(t, y, {"steady": 0, "transient": 1}, penalty=0, **kw)
    assert len(bic.segments) == 2  # the data has two levels: "at most" stays "at most"
    assert len(filled.segments) == 5  # with no penalty, every slot is used
    assert bic.penalty["steady"] > 0


def test_min_duration_and_transitions_are_respected():
    rng = np.random.default_rng(4)
    t = np.arange(200.0)
    y = np.where(t < 100, 0.0, 6.0) + rng.normal(0, 0.1, 200)
    alt = {"steady": ["transient"], "transient": ["steady"]}
    r = optimal_segments(t, y, {"steady": 0, "transient": 1}, method="exact", grid_s=2,
                         min_duration_s={"steady": 40.0, "transient": 20.0}, transitions=alt)
    states = [s.state.value for s in r.segments]
    assert all(b in alt[a] for a, b in zip(states, states[1:]))
    for s in r.segments:
        d = s.stop - s.start
        assert d >= (40 if s.state == ST else 20)


def test_alternation_changes_the_answer_only_where_it_should():
    rng = np.random.default_rng(5)
    t = np.arange(300.0)
    models = {"steady": 0, "transient": 1}
    kw = dict(method="exact", grid_s=1, max_counts={"steady": 2, "transient": 1},
              penalty={"steady": 0.5, "transient": 5.0}, min_duration_s={"transient": 30.0})
    alt = {"steady": ["transient"], "transient": ["steady"]}
    # A sharp step: unrestricted, two steady fits with a jump between them.
    step = np.where(t < 150, 0.0, 10.0) + rng.normal(0, 0.1, 300)
    free = optimal_segments(t, step, models, **kw)
    forced = optimal_segments(t, step, models, transitions=alt, **kw)
    assert [s.state for s in free.segments] == [ST, ST] and free.segments[1].start == 150
    # Alternation changes the labels: no steady -> steady. It does not make the transient cover the step;
    # segments are fitted independently, so the jump moves to a transient/steady boundary instead.
    states = [s.state for s in forced.segments]
    assert TR in states and all(not (a == b == ST) for a, b in zip(states, states[1:]))
    assert 150 in [s.start for s in forced.segments]
    assert forced.objective >= free.objective
    # A slow ramp between levels: steady - transient - steady either way.
    ramp = np.clip((t - 100) / 100, 0, 1) * 10 + rng.normal(0, 0.1, 300)
    free = optimal_segments(t, ramp, models, **kw)
    forced = optimal_segments(t, ramp, models, transitions=alt, **kw)
    assert [s.state for s in free.segments] == [ST, TR, ST]
    assert free.objective == pytest.approx(forced.objective)


# -- methods ------------------------------------------------------------------------


def _startup_record(seed=6, n=1800):
    rng = np.random.default_rng(seed)
    t = np.arange(float(n))
    y = np.zeros(n)
    up, down = int(n * 2 / 9), int(n * 13 / 18)
    y[up:] = 100 * (1 - np.exp(-(t[up:] - up) / 60.0))
    y[down:] = y[down - 1] * np.exp(-(t[down:] - down) / 30.0)
    return t, y + rng.normal(0, 0.5, n), (up, down)


def test_subset_refine_exact_never_get_worse():
    t, y, (up, down) = _startup_record()
    models = {"steady": 0, "transient": 1}
    cands = [0, up + 37, up + 300, down - 41, down + 150, len(y)]  # detector-like: late and imprecise
    kw = dict(max_counts={"steady": 3, "transient": 2}, penalty=10.0)
    sub = optimal_segments(t, y, models, method="subset", candidates=cands, **kw)
    ref = optimal_segments(t, y, models, method="refine", candidates=cands, refine_s=120.0, **kw)
    ex = optimal_segments(t, y, models, method="exact", candidates=cands, **kw)
    assert sub.objective >= ref.objective >= ex.objective - 1e-9
    assert ref.objective < sub.objective  # refinement found better boundaries
    assert all(ref.counts["steady"] <= 3) and all(ref.counts["transient"] <= 2)


def test_output_feeds_the_reports_and_compares_with_the_detector():
    t, y, _ = _startup_record()
    models = {"steady": 0, "transient": {"model": "exp", "T_min": 5.0, "T_max": 1e4}}
    cands = list(range(0, len(y), 100)) + [len(y)]
    r = optimal_segments(t, y, models, method="refine", candidates=cands, refine_s=60.0,
                         max_counts={"steady": 3, "transient": 2}, max_duration_s={"transient": 900.0})
    assert sum(len(s) for s in r.segments) == len(y)
    rep = rs.report.fit_report({"power": y}, r.segments, {"power": models}, t=t)
    assert rep.by_state.loc[("power", "all"), "n"] == len(y)
    assert rep.by_state.loc[("power", "all"), "rmse"] == pytest.approx(r.rms, rel=1e-9)
    # Agreement with another segmentation, sample by sample.
    det = rs.Detector([rs.Signal("p", "p")], [rs.DeviationCriterion("p", 3.0, 0.5)],
                      timing=rs.Timing(block_s=10, window_s=60))
    out = rs.batch.run(det, pd.DataFrame({"p": y}), t=t)
    cmp = compare_segments(r.segments, rs.segments_from(det, out), len(y))
    assert 0.0 <= cmp["agreement"] <= 1.0 and cmp["confusion"].to_numpy().sum() == len(y)
    assert compare_segments(r.segments, r.segments, len(y))["agreement"] == 1.0
    assert "segments" in r.summary() and "RMS" in str(r)


def test_too_short_for_the_model_is_never_chosen():
    t, y, _ = _startup_record()
    models = {"steady": 0, "transient": "exp"}  # exp needs at least 5 samples
    cands = [0, 400, 402, 404, 700, 1300, 1302, 1500, len(y)]
    r = optimal_segments(t, y, models, method="subset", candidates=cands, max_counts={"transient": 2})
    assert all(len(s) >= 5 for s in r.segments if s.state == TR)


def test_validation():
    t = np.arange(10.0)
    y = np.zeros(10)
    with pytest.raises(ValueError, match="method must be one of"):
        optimal_segments(t, y, {"steady": 0}, method="greedy")
    with pytest.raises(ValueError, match="needs candidates"):
        optimal_segments(t, y, {"steady": 0}, method="subset")
    with pytest.raises(ValueError, match="models names 'warm', which is not a state"):
        optimal_segments(t, y, {"warm": 0}, method="exact")
    with pytest.raises(ValueError, match="max_counts names state 'transient'"):
        optimal_segments(t, y, {"steady": 0}, method="exact", max_counts={"transient": 1})
    with pytest.raises(ValueError, match="model for 'y' in state 'steady'"):
        optimal_segments(t, y, {"steady": {"model": "poly", "degree": 2}}, method="exact")
    with pytest.raises(ValueError, match="no segmentation satisfies"):
        optimal_segments(t, y, {"steady": 0}, method="exact", max_counts={"steady": 0})
    with pytest.raises(ValueError, match="needs times as a DatetimeIndex"):
        optimal_segments(t, y, {"steady": 0}, method="exact", per="1D")
    with pytest.raises(ValueError, match="strictly increasing"):
        optimal_segments(t[::-1], y, {"steady": 0}, method="exact")


def test_clock_origin_does_not_matter():
    t, y, (up, down) = _startup_record(n=900)
    models = {"steady": 0, "transient": 1}
    kw = dict(method="exact", grid_s=10, per=300.0, max_counts={"steady": 2, "transient": 2}, penalty=5.0)
    a = optimal_segments(t, y, models, **kw)
    b = optimal_segments(t + 3e8, y, models, period_origin=3e8, **kw)
    assert [(s.start, s.stop, s.state) for s in a.segments] == [(s.start, s.stop, s.state) for s in b.segments]
    assert b.objective == pytest.approx(a.objective, rel=1e-6)


# -- review regressions (PR #46) -------------------------------------------------------


@pytest.mark.parametrize("order", [2, 3])
def test_pruning_waits_for_the_full_polynomial_order(order):
    """fit_poly reduces the order of short segments, so a longer one can fit much better."""
    t = np.arange(order + 2.0)  # the fewest samples at which fit_poly uses the full order
    kw = dict(method="exact", max_counts={"steady": 1}, penalty=0)
    r = optimal_segments(t, t ** order, {"steady": order}, **kw)
    assert r.sse == pytest.approx(0.0, abs=1e-12) and len(r.segments) == 1
    rng = np.random.default_rng(order)
    t = np.arange(60.0)
    y = np.where(t < 30, 0.01 * (t - 15) ** 2, 5 - 0.002 * (t - 45) ** 3) + rng.normal(0, 0.05, 60)
    models = {"steady": order, "transient": {"model": "exp", "T_min": 1.0, "T_max": 1e3}}
    for kw in (dict(method="exact", grid_s=3), dict(method="subset", candidates=[0, 10, 20, 28, 31, 40, 50, 60])):
        a = optimal_segments(t, y, models, max_counts={"steady": 2, "transient": 1}, prune=True, **kw)
        b = optimal_segments(t, y, models, max_counts={"steady": 2, "transient": 1}, prune=False, **kw)
        assert a.objective == pytest.approx(b.objective, rel=1e-12)


def test_fast_costs_are_stable_across_large_level_changes():
    r = optimal_segments(np.arange(4.0), np.array([1e9, 1.0, 2.0, 3.0]), {"steady": 0}, method="exact",
                         max_counts={"steady": 2}, penalty=0)
    assert [(s.start, s.stop) for s in r.segments] == [(0, 1), (1, 4)]
    assert r.sse == pytest.approx(2.0) and r.rms == pytest.approx(math.sqrt(0.5))
    from rocksteady.optimal import _Costs

    rng = np.random.default_rng(7)
    n = 500
    t = np.arange(float(n))
    y = np.where(t % 100 < 50, 1e9, 0.0) + rng.normal(0, 1e-3, n)  # 12 orders of magnitude apart
    y[[3, 260]] = np.nan
    for order in (0, 1):
        c = _Costs(t, y, {"steady": order}, ["steady"])
        for p, q in [(0, 50), (50, 100), (60, 70), (149, 152), (150, 200), (250, 270), (40, 60), (0, 500)]:
            fitted = fit_poly(t[p:q], y[p:q], order)
            e = fitted.residuals[np.isfinite(fitted.residuals)]
            want = float(e @ e)
            got = c.cost("steady", p, q)
            assert got == pytest.approx(want, rel=1e-6, abs=1e-6 * max(1.0, want)), (order, p, q, got, want)
        # Vectorised over starts: the same values.
        ps = np.arange(100, 199)
        v = c.vector("steady", ps, 200)
        for p, got in zip(ps[::7], v[::7]):
            e = fit_poly(t[p:200], y[p:200], order).residuals
            e = e[np.isfinite(e)]
            assert got == pytest.approx(float(e @ e), rel=1e-6, abs=1e-12)


def test_range_sums_bound_their_error():
    from rocksteady.optimal import _RangeSum

    rng = np.random.default_rng(8)
    x = np.concatenate((rng.normal(0, 1, 300) * 1e12, rng.normal(0, 1, 300), rng.normal(5, 1e-6, 300)))
    rs_ = _RangeSum(x)
    for _ in range(300):
        p, q = sorted(rng.integers(0, len(x) + 1, 2))
        v, err = rs_.range(int(p), int(q))
        assert abs(v - math.fsum(x[p:q])) <= err + 1e-300


def test_count_limits_hold_across_a_dst_fall_back():
    t = pd.date_range("2026-11-01 00:30", periods=8, freq="30min", tz="America/Chicago")  # 01:00-02:00 twice
    r = optimal_segments(t, np.arange(8.0), {"steady": 0}, method="exact", per="30min",
                         max_counts={"steady": 1}, penalty=0)
    assert r.counts["steady"].max() <= 1
    assert count_by_period(r.segments, t, "30min")["steady"].max() <= 1
    assert r.counts.index.is_monotonic_increasing and r.counts.index.is_unique
    # Days are unaffected by the fall-back and the spring-forward days.
    for start in ("2026-10-31", "2026-03-07"):
        idx = pd.date_range(start, periods=3 * 24 * 6, freq="10min", tz="America/Chicago")
        segs = [Segment(0, len(idx), ST)]
        assert count_by_period(segs, idx, "1D")["steady"].tolist() == [1] * len(
            pd.unique(idx.tz_localize(None).normalize()))


def test_linear_costs_survive_a_large_time_gap():
    """A gap elsewhere can round a short segment's time variance to zero; the slope must not be dropped."""
    t = np.array([0.0, 0.001, 0.002, 0.003, 1e6])
    y = np.array([0.0, 1.0, 2.0, 3.0, 0.0])
    r = optimal_segments(t, y, {"transient": 1}, method="exact", max_counts={"transient": 2}, penalty=0)
    assert [(s.start, s.stop) for s in r.segments] == [(0, 4), (4, 5)]
    assert r.sse == pytest.approx(0.0, abs=1e-12)
    from rocksteady.optimal import _Costs

    rng = np.random.default_rng(10)
    t = np.concatenate((1e-3 * np.arange(50), 1e7 + np.arange(50.0)))  # a 1e7 s gap; 1 ms spacing before it
    y = np.concatenate((2.0 + 3e3 * t[:50], 5.0 - 0.1 * (t[50:] - 1e7))) + rng.normal(0, 1e-3, 100)
    c = _Costs(t, y, {"transient": 1}, ["transient"])
    for p, q in [(0, 3), (0, 10), (5, 40), (0, 50), (50, 60), (45, 55), (0, 100)]:
        e = fit_poly(t[p:q], y[p:q], 1).residuals
        e = e[np.isfinite(e)]
        assert c.cost("transient", p, q) == pytest.approx(float(e @ e), rel=1e-6, abs=1e-9), (p, q)


# -- infeasibility (#62) -----------------------------------------------------------------


def _infeasible_message(*args, **kwargs) -> str:
    with pytest.raises(ValueError, match="no segmentation satisfies") as e:
        optimal_segments(*args, **kwargs)
    return str(e.value)


def test_an_infeasible_day_is_named_with_the_time():
    # From noon on day 1 to the end of day 2, hourly. A segment lasts at most 20 h, and at most one may touch
    # each day: day 1 (12 h) can be covered, day 2 (24 h) cannot. Every segmentation gets stuck on day 2, at
    # 20:00 at the latest (a segment from 00:00 that lasts the full 20 h).
    t = pd.date_range("2026-01-01 12:00", "2026-01-02 23:00", freq="1h")
    y = np.zeros(len(t))
    msg = _infeasible_message(t, y, {"steady": 0}, method="exact", per="1D", max_counts={"steady": 1},
                              max_duration_s={"steady": 20 * 3600.0}, penalty=0.0)
    day2 = count_by_period([Segment(0, len(t), ST)], t, "1D").index[1]
    assert "2026-01-02 20:00:00" in msg
    assert f"period starting {day2}" in msg
    assert "2026-01-01" not in msg
    assert "max_counts={'steady': 1}" in msg and "transitions=any" in msg


def test_an_infeasible_record_without_periods_is_named_whole():
    # One segment of at most 4 s cannot cover 10 s: every segmentation gets stuck at 104 s, the caller's time.
    t = 100.0 + np.arange(10.0)
    msg = _infeasible_message(t, np.zeros(10), {"steady": 0}, method="exact", max_counts={"steady": 1},
                              max_duration_s={"steady": 4.0}, penalty=0.0)
    assert "continue from 104.0 s" in msg
    assert msg.count("the whole record") == 1 and "period starting" not in msg
    assert "max_counts={'steady': 1}" in msg


def test_a_duration_no_candidate_meets_is_named_the_same_way():
    # The candidate segments last 5 s or 10 s, none between 6 s and 8 s: nothing can start at the first sample.
    t = 3.0 + np.arange(10.0)
    msg = _infeasible_message(t, np.zeros(10), {"steady": 0}, method="subset", candidates=[0, 5, 10], per=5.0,
                              min_duration_s={"steady": 6.0}, max_duration_s={"steady": 8.0})
    assert re.search(r"none can continue from 3\.0 s \(sample 0\), in the period starting 0\.0 s\. "
                     r"In force: max_counts=none, transitions=any\.$", msg), msg


@pytest.mark.parametrize("prune", [True, False])
def test_the_time_is_the_models_not_the_pruning_surrogates(prune):
    # exp needs 5 samples, and no segment may last more than 3 s: none can start at all. The constant-fit
    # surrogate that pruning starts from accepts short segments, which must not move the time.
    t = np.arange(20.0)
    msg = _infeasible_message(t, np.zeros(20), {"transient": "exp"}, method="exact", prune=prune,
                              max_counts={"transient": 1}, max_duration_s={"transient": 3.0})
    assert "continue from 0.0 s (sample 0)" in msg


def _fit_ends(monkeypatch):
    """The last time of every segment the model fits are asked for."""
    ends, real = [], rs.fit.fit_model

    def spy(spec, t, y, state):
        ends.append(float(t[-1]))
        return real(spec, t, y, state)

    monkeypatch.setattr(rs.fit, "fit_model", spy)
    return ends


def test_an_infeasible_record_is_not_fitted_past_where_it_gets_stuck(monkeypatch):
    # One transient of at most 10 s and one steady segment of at most 200 s: stuck at 210 s of 1000 s. Pruning
    # starts from a constant-fit surrogate that fails as well; no segment after that point needs a model fit.
    t = np.arange(1000.0)
    ends = _fit_ends(monkeypatch)
    msg = _infeasible_message(t, np.zeros(1000), {"steady": 0, "transient": "exp"}, method="exact", grid_s=5.0,
                              max_counts={"steady": 1, "transient": 1},
                              max_duration_s={"steady": 200.0, "transient": 10.0}, penalty=0.0)
    assert "continue from 210.0 s" in msg
    assert ends and max(ends) < 210.0


def test_a_state_capped_at_zero_is_never_fitted(monkeypatch):
    # The probe of review 1 on #82: a transient capped at 0 cannot be placed, so its exp model is never fitted.
    t = np.arange(300.0)
    y = np.where(t < 100, 0.0, 5.0 * (1 - np.exp(-(t - 100) / 20))) + np.random.default_rng(0).normal(0, 0.05, 300)
    ends = _fit_ends(monkeypatch)
    msg = _infeasible_message(t, y, {"steady": 0, "transient": "exp"}, method="exact", grid_s=2.0, penalty=0.0,
                              max_counts={"transient": 0, "steady": 1}, max_duration_s={"steady": 200.0})
    assert "continue from 200.0 s" in msg
    assert ends == []


def test_the_constraints_in_force_read_as_plain_values():
    # NumPy caps, a Timedelta period and set-valued transitions print as the values they hold.
    t = pd.date_range("2026-01-01", periods=48, freq="1h")
    msg = _infeasible_message(t, np.zeros(48), {"steady": 0, "transient": 1}, method="exact",
                              per=pd.Timedelta("1D"), max_counts={"steady": np.int64(1), "transient": np.int64(0)},
                              transitions={"steady": {"transient", "steady"}, "transient": ("steady",)},
                              max_duration_s={"steady": 3600.0 * 20}, penalty=0.0)
    assert ("In force: max_counts={'steady': 1, 'transient': 0} per period (1 days 00:00:00), "
            "transitions={'steady': ['steady', 'transient'], 'transient': ['steady']}.") in msg, msg


# -- subtypes (#61) ----------------------------------------------------------------------


def _two_plateaus(seed=20, n=60):
    """Cold (0) for the first third, a ramp, then hot (5)."""
    rng = np.random.default_rng(seed)
    t = np.arange(float(n))
    y = np.interp(t, [0, 20, 30, n], [0.0, 0.0, 5.0, 5.0]) + rng.normal(0, 0.05, n)
    return t, y


def test_each_steady_segment_takes_its_level_subtype():
    t, y = _two_plateaus()
    subtypes = {"cold": rs.Subtype("steady", level=(None, 2.5)), "hot": rs.Subtype("steady", level=(2.5, None))}
    r = optimal_segments(t, y, {"steady": 0, "transient": 1}, method="exact", subtypes=subtypes, penalty=1.0,
                         transitions={"steady": ["transient"], "transient": ["steady"]})
    assert [(s.state, s.subtype) for s in r.segments] == [(ST, "cold"), (TR, None), (ST, "hot")]


def test_counts_and_summary_show_subtypes():
    t, y = _two_plateaus()
    subtypes = {"cold": rs.Subtype("steady", level=(None, 2.5)), "hot": rs.Subtype("steady", level=(2.5, None)),
                "warm": rs.Subtype("steady", level=(10.0, None))}
    r = optimal_segments(t, y, {"steady": 0, "transient": 1}, method="exact", subtypes=subtypes, penalty=1.0,
                         transitions={"steady": ["transient"], "transient": ["steady"]})
    assert r.subtypes == subtypes and list(r.counts.columns) == ["steady", "transient", "cold", "hot", "warm"]
    assert r.counts.iloc[0].tolist() == [2, 1, 1, 1, 0]
    assert "3 segments (steady 2: cold 1, hot 1, warm 0; transient 1)" in r.summary()
    plain = optimal_segments(t, y, {"steady": 0, "transient": 1}, method="exact", penalty=1.0,
                             transitions={"steady": ["transient"], "transient": ["steady"]})
    assert "(steady 2, transient 1)" in plain.summary() and list(plain.counts.columns) == ["steady", "transient"]


def _subtype_scheme(scheme, rng):
    """Random level subtypes: a split, an overlap, a catch-all, a gap, and subtypes of the transient state."""
    a = float(rng.uniform(0.5, 2.5))
    if scheme == "split":
        return {"cold": rs.Subtype("steady", level=(None, a)), "hot": rs.Subtype("steady", level=(a, None))}
    if scheme == "overlap":
        return {"hot": rs.Subtype("steady", level=(a - 0.6, None)), "cold": rs.Subtype("steady", level=(None, a))}
    if scheme == "catch-all":
        return {"hot": rs.Subtype("steady", level=(a, None)), "other": rs.Subtype("steady")}
    if scheme == "peak":
        return {"high": rs.Subtype("transient", peak=(a + 0.5, None)), "low": rs.Subtype("transient", peak=(None, a)),
                "other": rs.Subtype("transient"), "hot": rs.Subtype("steady", peak=(a, None)),
                "cold": rs.Subtype("steady")}
    if scheme == "rise":
        return {"fast": rs.Subtype("transient", max_rise_s=float(rng.integers(0, 3)),
                                   rise_fraction=float(rng.choice([0.5, 0.95, 1.0]))),
                "other": rs.Subtype("transient"), "hot": rs.Subtype("steady", level=(a, None)),
                "cold": rs.Subtype("steady")}
    if scheme == "pulse":  # all three conditions at once
        return {"pulse": rs.Subtype("transient", level=(None, a + 1.0), peak=(a, None), max_rise_s=1.0),
                "other": rs.Subtype("transient"), "hot": rs.Subtype("steady", level=(a, None)),
                "cold": rs.Subtype("steady")}
    if scheme == "gap":
        return {"cold": rs.Subtype("steady", level=(None, a - 0.4)), "hot": rs.Subtype("steady", level=(a + 0.4, None))}
    return {"low": rs.Subtype("transient", level=(None, a)), "any": rs.Subtype("transient"),
            "hot": rs.Subtype("steady", level=(a, None)), "cold": rs.Subtype("steady", level=(None, a))}


def _independent_objective(t, y, models, result, penalty):
    """The result's objective, from its segments and the reference fits."""
    return sum(_sse(models[s.state.value], t[s.slice], y[s.slice], s.state.value) + penalty for s in result.segments)


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("scheme", ["split", "overlap", "catch-all", "gap", "transient", "peak", "rise", "pulse"])
@pytest.mark.parametrize("periods", [False, True])
def test_exact_matches_brute_force_with_subtypes(seed, scheme, periods):
    rng = np.random.default_rng(100 + seed)
    n = 7
    t = np.arange(n, dtype=float)
    y = np.where(t < rng.integers(2, 5), 0.0, 3.0) + rng.normal(0, 0.3, n)
    if seed == 2:
        y[3] = np.nan  # a non-finite sample: levels use the finite samples
    models = {"steady": 0, "transient": 1}
    subtypes = _subtype_scheme(scheme, rng)
    caps = {"steady": 2, "hot": 1}  # a cap on a state and on one of its subtypes
    if scheme in ("transient", "peak"):
        caps["low"] = 1
    kw = dict(per=3.0, max_counts=caps) if periods else dict(max_counts=caps)
    want_cost, (_, want_names) = _brute(t, y, models, penalty=0.5, subtypes=subtypes, **kw)
    got = optimal_segments(t, y, models, method="exact", penalty=0.5, subtypes=subtypes, **kw)
    assert got.objective == pytest.approx(want_cost, rel=1e-9, abs=1e-9)
    assert _independent_objective(t, y, models, got, 0.5) == pytest.approx(got.objective, rel=1e-9, abs=1e-9)
    # The result meets its subtypes' conditions and every cap.
    for s in got.segments:
        assert s.subtype in [k for k, v in subtypes.items() if v.state == s.state] or (
            s.subtype is None and not any(v.state == s.state for v in subtypes.values()))
        assert s.subtype is None or _meets_conditions(subtypes[s.subtype], t[s.slice], y[s.slice])
    for name, cap in caps.items():
        assert (got.counts[name] <= cap).all()


@pytest.mark.parametrize("cap", [None, "high", "any"])
def test_an_exact_tie_goes_to_the_subtype_declared_first(cap):
    """Subtypes of one state have the same cost, so a segment both accept is a tie, capped or not."""
    t = np.arange(20.0)
    y = 3.0 + np.random.default_rng(30).normal(0, 0.1, 20)
    kw = dict(method="exact", penalty=100.0, max_counts={cap: 1} if cap else None)
    high, any_ = rs.Subtype("steady", level=(2.0, None)), rs.Subtype("steady")
    first = optimal_segments(t, y, {"steady": 0}, subtypes={"high": high, "any": any_}, **kw)
    assert [s.subtype for s in first.segments] == ["high"]
    first = optimal_segments(t, y, {"steady": 0}, subtypes={"any": any_, "high": high}, **kw)
    assert [s.subtype for s in first.segments] == ["any"]


@pytest.mark.parametrize("per", [None, 10.0])
def test_an_exact_tie_before_the_last_segment_goes_to_the_subtype_declared_first(per):
    t = np.arange(20.0)
    y = np.where(t < 10, 3.0, 0.0) + np.random.default_rng(31).normal(0, 0.1, 20)
    high, any_ = rs.Subtype("steady", level=(2.0, None)), rs.Subtype("steady")
    r = optimal_segments(t, y, {"steady": 0}, method="exact", penalty=1.0, per=per, max_counts={"high": 1},
                         subtypes={"high": high, "any": any_})
    assert [(s.start, s.subtype) for s in r.segments] == [(0, "high"), (10, "any")]


def test_no_cap_can_be_dodged():
    """Cold and hot cover every level; with one hot segment a day, no day has two hot-level steady segments."""
    rng = np.random.default_rng(40)
    idx = pd.date_range("2026-03-02 00:00", periods=2 * 24 * 6, freq="10min")
    hour = idx.hour + idx.minute / 60.0
    hot = ((hour >= 8) & (hour < 11)) | ((hour >= 14) & (hour < 17))  # two hot plateaus a day
    y = np.where(hot, 100.0, 0.0) + rng.normal(0, 1.0, len(idx))
    subtypes = {"cold": rs.Subtype("steady", level=(None, 50.0)), "hot": rs.Subtype("steady", level=(50.0, None))}
    models = {"steady": 0, "transient": 1}
    kw = dict(method="exact", grid_s=1800, per="1D", penalty={"steady": 50.0, "transient": 1e6})
    free = optimal_segments(idx, y, models, subtypes=subtypes, **kw)
    assert (free.counts["hot"] == 2).all()  # the cap binds
    r = optimal_segments(idx, y, models, subtypes=subtypes, max_counts={"hot": 1}, **kw)
    day = idx.normalize()
    for d in day.unique():
        n_hot = sum(1 for s in r.segments if s.state == ST and day[s.start] <= d <= day[s.stop - 1]
                    and np.nanmean(y[s.slice]) >= 50.0)
        assert n_hot <= 1
    assert all(s.subtype in ("cold", "hot") for s in r.segments if s.state == ST)
    assert (r.counts["hot"] <= 1).all() and r.objective > free.objective


@pytest.mark.parametrize("y, at", [([40.0] * 6, 40.0), ([39.5, 40.5] * 3, 40.0),
                                   ([1e9, 1e9, 40.0, 40.0, 40.0, 40.0], 40.0),
                                   ([0.7] * 53, 0.7), ([97.7379777971513] * 11, 97.7379777971513),
                                   ([0.1, 0.3, 0.2], 0.2)])
def test_a_mean_exactly_at_a_threshold_takes_the_upper_subtype(y, at):
    """The exact mean: ``fsum(y) / n`` rounds twice and falls below 0.7 for 53 samples of 0.7."""
    y = np.array(y)
    t = np.arange(float(len(y)))
    subtypes = {"cold": rs.Subtype("steady", level=(None, at)), "hot": rs.Subtype("steady", level=(at, None))}
    cands = [0, 2, len(y)] if y[0] == 1e9 else [0, len(y)]
    r = optimal_segments(t, y, {"steady": 0}, method="subset", candidates=cands, penalty=0, subtypes=subtypes)
    want = "hot" if _exact_level_ok(y[r.segments[-1].slice], (at, None)) else "cold"
    assert r.segments[-1].subtype == want
    assert want == ("cold" if at == 0.2 else "hot")  # 0.1 + 0.3 + 0.2 is exactly below 0.6


def test_refine_keeps_every_segment_within_its_level():
    t, y, (up, down) = _startup_record()
    models = {"steady": 0, "transient": 1}
    subtypes = {"zero": rs.Subtype("steady", level=(None, 5.0)), "power": rs.Subtype("steady", level=(5.0, None))}
    cands = [0, up + 37, up + 300, down - 41, down + 150, len(y)]
    kw = dict(candidates=cands, max_counts={"steady": 3, "transient": 2, "power": 1}, penalty=10.0, subtypes=subtypes)
    sub = optimal_segments(t, y, models, method="subset", **kw)
    for refine_s in (30.0, 120.0, 400.0):
        ref = optimal_segments(t, y, models, method="refine", refine_s=refine_s, **kw)
        assert ref.objective <= sub.objective
        for s in ref.segments:
            if s.state == ST:
                level = subtypes[s.subtype].level
                m = np.nanmean(y[s.slice])
                assert (level[0] is None or m >= level[0]) and (level[1] is None or m < level[1])
    assert [s.subtype for s in ref.segments if s.state == ST] == [s.subtype for s in sub.segments if s.state == ST]


def test_pruning_stays_exact_with_subtypes():
    """Several options per state, conditions that reject segments, and a cap on a state and on its subtype."""
    t, y, (up, down) = _startup_record(n=900)
    models = {"steady": 0, "transient": {"model": "first_order", "tau_min": 1.0, "tau_max": 1e3}}
    subtypes = {"zero": rs.Subtype("steady", level=(None, 5.0)), "power": rs.Subtype("steady", level=(60.0, None)),
                "rising": rs.Subtype("transient", level=(None, 50.0)), "other": rs.Subtype("transient")}
    for kw in (dict(method="exact", grid_s=30), dict(method="subset", candidates=list(range(0, 901, 25)))):
        a, b = (optimal_segments(t, y, models, subtypes=subtypes, per=450.0, prune=prune,
                                 max_counts={"steady": 2, "power": 1, "transient": 2}, **kw) for prune in (True, False))
        assert a.objective == pytest.approx(b.objective, rel=1e-12)
        assert [(s.start, s.stop, s.subtype) for s in a.segments] == [(s.start, s.stop, s.subtype) for s in b.segments]
        if kw["method"] == "exact":
            assert {s.subtype for s in a.segments} >= {"zero", "power", "rising", "other"}


def test_no_subtypes_gives_the_same_result():
    t, y, (up, down) = _startup_record(n=900)
    kw = dict(method="exact", grid_s=10, per=300.0, max_counts={"steady": 2, "transient": 2}, penalty=5.0)
    a = optimal_segments(t, y, {"steady": 0, "transient": 1}, **kw)
    b = optimal_segments(t, y, {"steady": 0, "transient": 1}, subtypes={}, **kw)
    assert a.segments == b.segments and a.objective == b.objective and a.counts.equals(b.counts)
    assert all(s.subtype is None for s in a.segments)


def test_subtype_validation():
    t = np.arange(10.0)
    y = np.zeros(10)
    with pytest.raises(ValueError, match="Subtype state must be one of steady, transient, indeterminate; got 'warm'"):
        rs.Subtype("warm")
    assert rs.Subtype("steady", level=np.array([1.0, 2.0])).level == (1.0, 2.0)  # a pair as an array
    for level in [(5.0, 1.0), (1.0, 1.0)]:
        with pytest.raises(ValueError, match=r"level lo \(\S+\) must be < Subtype level hi"):
            rs.Subtype("steady", level=level)
    with pytest.raises(ValueError, match="level must be a pair"):
        rs.Subtype("steady", level=40.0)
    with pytest.raises(ValueError, match="level lo must be finite"):
        rs.Subtype("steady", level=(float("nan"), None))
    with pytest.raises(dataclasses.FrozenInstanceError):
        rs.Subtype("steady").level = (0, 1)

    def run(subtypes, **kw):
        return optimal_segments(t, y, {"steady": 0}, method="exact", subtypes=subtypes, **kw)

    with pytest.raises(ValueError, match="must be a Subtype"):
        run({"hot": ("steady", (40, None))})
    for name in ("", 3):
        with pytest.raises(ValueError, match="a subtype's name must be a non-empty string"):
            run({name: rs.Subtype("steady")})
    for name in ("steady", "transient"):
        with pytest.raises(ValueError, match=f"subtype '{name}' has the name of a state"):
            run({name: rs.Subtype("steady")})
    with pytest.raises(ValueError, match="subtype 'startup' refines state 'transient', which has no model"):
        run({"startup": rs.Subtype("transient")})
    with pytest.raises(ValueError, match="max_counts names 'warm', which is not a state with a model or a subtype"):
        run({"hot": rs.Subtype("steady")}, max_counts={"warm": 1})
    with pytest.raises(ValueError, match=r"max_counts\['hot'\] must be a non-negative integer"):
        run({"hot": rs.Subtype("steady")}, max_counts={"hot": -1})
    with pytest.raises(ValueError, match="penalty names subtype 'hot'; it takes states only"):
        run({"hot": rs.Subtype("steady")}, penalty={"hot": 5.0})
    with pytest.raises(ValueError, match="subtypes' conditions"):
        run({"hot": rs.Subtype("steady", level=(100.0, None))})
    with pytest.raises(ValueError, match=r"max_counts=\{'hot': 0\} per period \(5.0\)"):  # a subtype's cap
        run({"hot": rs.Subtype("steady")}, max_counts={"hot": 0}, per=5.0)


def test_a_tie_between_predecessors_of_different_states_still_goes_to_the_subtype_declared_first():
    """Review 1 of PR #83: [3, 5) took 'mid' although 'low', declared first, accepts it at the same objective."""
    t = np.arange(5.0)
    y = np.array([2.0, 1.0, 0.0, 2.0, 1.0])
    subtypes = {"low": rs.Subtype("steady", level=(None, 1.8)), "mid": rs.Subtype("steady", level=(1.4, 1.8)),
                "any": rs.Subtype("steady")}
    r = optimal_segments(t, y, {"steady": 0, "transient": 1}, subtypes=subtypes, method="exact", penalty=0.5)
    assert [(s.start, s.stop, s.subtype) for s in r.segments][-1] == (3, 5, "low")


def _exact_level_ok(y, level):
    """The level condition by exact arithmetic: the mean of the finite samples in [lo, hi)."""
    from fractions import Fraction

    v = [Fraction(float(x)) for x in y if np.isfinite(x)]
    if not v or level is None:
        return bool(v) or level is None
    m = sum(v) / len(v)
    return (level[0] is None or m >= Fraction(level[0])) and (level[1] is None or m < Fraction(level[1]))


def _within(events, sl):
    """Each event mask's part within the samples ``sl``."""
    return {k: m[sl] for k, m in (events or {}).items()}


def _meets_conditions(sub, t, y, events=None):
    """
    A segment's samples ``y`` at times ``t``, and its part of each event mask (``events``), against every
    condition of ``sub``, sample by sample.
    """
    for name, want in sub.events or ():
        if any(bool(f) for f in events[name]) != want:
            return False
    finite = [i for i in range(len(y)) if np.isfinite(y[i])]
    if sub.level is not None and not _exact_level_ok(y, sub.level):
        return False
    if sub.peak is None and sub.max_rise_s is None:
        return True
    if not finite:
        return False
    top = max(y[i] for i in finite)
    if sub.peak is not None:
        lo, hi = sub.peak
        if not ((lo is None or top >= lo) and (hi is None or top < hi)):
            return False
    if sub.max_rise_s is not None:
        y0 = y[finite[0]]
        thr = min(y0 + sub.rise_fraction * (top - y0), top)
        first = next(i for i in finite if y[i] >= thr)
        if t[first] - t[0] > sub.max_rise_s:
            return False
    return True


@pytest.mark.parametrize("seed", range(40))
def test_every_segment_takes_the_first_declared_subtype_the_caps_allow(seed):
    """
    Ties are broken by the sum of the subtypes' positions, so no segment could take an earlier-declared
    subtype that accepts it without breaking a cap. Quantised data makes exact ties common.
    """
    rng = np.random.default_rng(500 + seed)
    n = int(rng.integers(4, 9))
    t = np.arange(float(n))
    y = rng.integers(0, 3, n).astype(float)
    subtypes = {"low": rs.Subtype("steady", level=(None, 1.8)), "mid": rs.Subtype("steady", level=(0.5, 1.8)),
                "any": rs.Subtype("steady"), "up": rs.Subtype("transient", level=(1.0, None)),
                "other": rs.Subtype("transient")}
    caps = {"mid": 1, "up": 1} if seed % 2 else None
    r = optimal_segments(t, y, {"steady": 0, "transient": 0}, subtypes=subtypes, method="exact",
                         penalty=float(rng.choice([0.0, 0.5])), max_counts=caps)
    used = [s.subtype for s in r.segments]
    for i, s in enumerate(r.segments):
        mine = [k for k, v in subtypes.items() if v.state == s.state]
        for earlier in mine[:mine.index(s.subtype)]:
            if not _exact_level_ok(y[s.slice], subtypes[earlier].level):
                continue
            swapped = used[:i] + [earlier] + used[i + 1:]
            assert caps is not None and any(swapped.count(k) > c for k, c in caps.items()), (i, s, earlier)


# -- peak and rise-time conditions (#64) ------------------------------------------------


def _subtype_of(t, y, subtypes, models=None):
    """The subtype the whole record takes as one segment."""
    r = optimal_segments(t, np.asarray(y, dtype=float), models or {"transient": 1}, method="subset",
                         candidates=[0, len(y)], penalty=0, subtypes=subtypes)
    return r.segments[0].subtype


@pytest.mark.parametrize("top, reached", [(5.0, True), (np.nextafter(5.0, 0.0), False), (np.nextafter(9.0, 0), True),
                                          (9.0, False)])
def test_a_peak_must_lie_in_its_range(top, reached):
    """The maximum of the segment's samples in [lo, hi): exactly at lo reaches it, just below does not; hi is open."""
    t = np.arange(6.0)
    y = [1.0, 2.0, top, 3.0, np.nan, 2.0]
    subtypes = {"startup": rs.Subtype("transient", peak=(5.0, 9.0)), "other": rs.Subtype("transient")}
    assert _subtype_of(t, y, subtypes) == ("startup" if reached else "other")


@pytest.mark.parametrize("max_rise_s, fast", [(3.0, True), (np.nextafter(3.0, 0.0), False), (3.5, True), (2.5, False)])
def test_a_rise_must_take_at_most_max_rise_s(max_rise_s, fast):
    """0 to a peak of 10: the first sample at or above 9.5 (0.95 of the way) is 3 s after the segment's first."""
    t = np.arange(6.0)
    y = [0.0, 2.0, 9.0, 9.5, 10.0, 8.0]
    subtypes = {"pulse": rs.Subtype("transient", max_rise_s=max_rise_s), "other": rs.Subtype("transient")}
    assert _subtype_of(t, y, subtypes) == ("pulse" if fast else "other")


@pytest.mark.parametrize("max_rise_s, fast", [(6.0, True), (5.5, False)])
def test_rise_time_is_measured_from_the_segments_start_and_starting_value(max_rise_s, fast):
    """
    A rise from 20 to 25 starting at t = 10: 0.95 of the way is 24.75, first reached at t = 16, so 6 s. Measured
    from zero the threshold would be 23.75 (t = 15, 5 s); measured from the record's start, 16 s.
    """
    t = np.arange(18.0)
    y = np.array([20.0] * 10 + [20.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 25.0])
    subtypes = {"pulse": rs.Subtype("transient", max_rise_s=max_rise_s), "other": rs.Subtype("transient")}
    r = optimal_segments(t, y, {"steady": 0, "transient": 1}, method="subset", candidates=[0, 10, 18], penalty=0,
                         transitions={"steady": ["transient"], "transient": []}, subtypes=subtypes)
    want = [(0, ST, None), (10, TR, "pulse" if fast else "other")]
    assert [(s.start, s.state, s.subtype) for s in r.segments] == want


def test_peak_and_rise_validation():
    assert rs.Subtype("transient", max_rise_s=10).rise_fraction == 0.95
    assert rs.Subtype("transient", max_rise_s=0.0, rise_fraction=1).rise_fraction == 1.0
    assert rs.Subtype("transient", peak=np.array([1.0, None])).peak == (1.0, None)
    for peak in [(5.0, 1.0), (1.0, 1.0)]:
        with pytest.raises(ValueError, match=r"peak lo \(\S+\) must be < Subtype peak hi"):
            rs.Subtype("transient", peak=peak)
    with pytest.raises(ValueError, match="peak must be a pair"):
        rs.Subtype("transient", peak=40.0)
    for bad in (-1.0, float("nan"), float("inf"), "10", True):
        with pytest.raises(ValueError, match="Subtype max_rise_s must be"):
            rs.Subtype("transient", max_rise_s=bad)
    for bad in (0.0, -0.5, 1.5, float("nan"), "0.9"):
        with pytest.raises(ValueError, match=r"Subtype rise_fraction must be in \(0, 1\]"):
            rs.Subtype("transient", max_rise_s=10.0, rise_fraction=bad)
    with pytest.raises(ValueError, match="rise_fraction needs max_rise_s"):
        rs.Subtype("transient", rise_fraction=0.9)


@pytest.mark.parametrize("seed", range(12))
def test_on_long_records_every_segment_takes_the_first_subtype_whose_peak_and_rise_it_meets(seed):
    """Segments of every length, within one block of samples and across many, with gaps in the data."""
    rng = np.random.default_rng(700 + seed)
    n = int(rng.integers(300, 1500))
    t = np.cumsum(rng.uniform(0.5, 1.5, n))
    y = np.cumsum(rng.normal(0, 1.0, n))
    y[rng.random(n) < 0.05] = np.nan
    if seed == 0:
        y[100:240] = np.nan  # a long gap: segments in it have no finite sample
    lo = float(np.nanquantile(y, 0.6))
    subtypes = {"pulse": rs.Subtype("transient", peak=(lo, None), max_rise_s=float(rng.uniform(5, 80)),
                                    rise_fraction=float(rng.uniform(0.3, 1.0))),
                "high": rs.Subtype("transient", peak=(lo, None)),
                "fast": rs.Subtype("transient", max_rise_s=float(rng.uniform(5, 200))),
                "other": rs.Subtype("transient")}
    cands = np.unique(np.concatenate(([0, 120, 200, n], rng.integers(1, n, int(rng.integers(5, 40))))))
    penalty = float(rng.choice([0.0, 300.0, 3000.0, 30000.0]))
    r = optimal_segments(t, y, {"transient": 0}, method="subset", candidates=cands, penalty=penalty, subtypes=subtypes)
    for s in r.segments:
        want = next(k for k, v in subtypes.items() if _meets_conditions(v, t[s.slice], y[s.slice]))
        assert s.subtype == want, (s, want)


@pytest.mark.parametrize("n", [700, 1000])
def test_a_peak_is_found_wherever_it_lies_in_a_long_segment(n):
    t = np.arange(float(n))
    base = 0.5 * np.sin(t / 17.0)
    subtypes = {"spiked": rs.Subtype("transient", peak=(2.0, None)), "other": rs.Subtype("transient")}
    for j in range(0, n, 7):
        y = base.copy()
        y[j] = 2.0
        assert _subtype_of(t, y, subtypes) == "spiked", j
    assert _subtype_of(t, base, subtypes) == "other"


def test_a_segment_with_no_finite_sample_meets_no_peak_or_rise_condition():
    t = np.arange(9.0)
    y = np.array([0.0, 1.0, 2.0, np.nan, np.inf, np.nan, 2.0, 1.0, 0.0])
    subtypes = {"below": rs.Subtype("transient", peak=(None, 100.0)), "fast": rs.Subtype("transient", max_rise_s=100.0),
                "other": rs.Subtype("transient")}
    r = optimal_segments(t, y, {"transient": 0}, method="subset", candidates=[0, 3, 6, 9], penalty=0,
                         max_duration_s={"transient": 3.0}, subtypes=subtypes)
    assert [s.subtype for s in r.segments] == ["below", "other", "below"]


def test_refine_keeps_every_segment_within_its_peak_and_rise():
    """Without the subtypes, refine would move the falling transient's start past the samples above 90."""
    t, y, (up, down) = _startup_record()
    models = {"steady": 0, "transient": 1}
    subtypes = {"zero": rs.Subtype("steady", level=(None, 5.0)), "power": rs.Subtype("steady", level=(5.0, None)),
                "high": rs.Subtype("transient", peak=(90.0, None), max_rise_s=60.0), "other": rs.Subtype("transient")}
    cands = [0, up + 37, up + 300, down - 41, down + 150, len(y)]
    kw = dict(candidates=cands, penalty=10.0)
    caps = {"steady": 3, "transient": 2}
    sub = optimal_segments(t, y, models, method="subset", max_counts=caps, subtypes=subtypes, **kw)
    assert [s.subtype for s in sub.segments] == ["zero", "other", "power", "high", "zero"]
    moved_out = False
    for refine_s in (30.0, 120.0, 400.0):
        ref = optimal_segments(t, y, models, method="refine", refine_s=refine_s, max_counts=caps, subtypes=subtypes,
                               **kw)
        assert ref.objective <= sub.objective
        assert [s.subtype for s in ref.segments] == [s.subtype for s in sub.segments]
        for s in ref.segments:
            assert _meets_conditions(subtypes[s.subtype], t[s.slice], y[s.slice])
        free = optimal_segments(t, y, models, method="refine", refine_s=refine_s, max_counts=caps, **kw)
        moved_out |= not all(_meets_conditions(subtypes[a.subtype], t[b.slice], y[b.slice])
                             for a, b in zip(sub.segments, free.segments))
    assert moved_out


# -- event conditions (#65) -------------------------------------------------------------


@pytest.mark.parametrize("flag, driven", [(4, 1), (7, 1), (3, 0), (8, 2)])
def test_a_segment_contains_a_flagged_sample_from_its_first_to_its_last_sample_only(flag, driven):
    """Segments [0, 4), [4, 8), [8, 12): a flag at 4 or 7 lies in the middle one, at 3 or 8 just outside it."""
    t = np.arange(12.0)
    heater = np.zeros(12, dtype=bool)
    heater[flag] = True
    subtypes = {"driven": rs.Subtype("transient", events={"heater": True}),
                "undriven": rs.Subtype("transient", events={"heater": False})}
    r = optimal_segments(t, np.sin(t), {"transient": 1}, method="subset", candidates=[0, 4, 8, 12], penalty=0,
                         subtypes=subtypes, events={"heater": heater})
    assert [s.subtype for s in r.segments] == ["driven" if i == driven else "undriven" for i in range(3)]


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("periods", [False, True])
def test_exact_matches_brute_force_with_events_on_two_masks(seed, periods):
    rng = np.random.default_rng(800 + seed)
    n = 7
    t = np.arange(n, dtype=float)
    y = np.where(t < rng.integers(2, 5), 0.0, 3.0) + rng.normal(0, 0.3, n)
    events = {"heater": rng.random(n) < 0.25, "valve": rng.random(n) < 0.35}
    if seed == 5:  # a flag on a non-finite sample: an event condition reads the mask, whatever the signal there
        y[2] = np.nan
        events["heater"][2] = True
    a = float(rng.uniform(0.5, 2.5))
    subtypes = {"heat_up": rs.Subtype("transient", events={"heater": True}),
                "quiet": rs.Subtype("transient", events={"heater": False, "valve": False}),
                "valved": rs.Subtype("steady", level=(a, None), events={"valve": True}),
                "steady_other": rs.Subtype("steady")}
    caps = {"heat_up": 1, "valved": 1, "steady": 2}
    kw = dict(per=3.0, max_counts=caps) if periods else dict(max_counts=caps)
    want_cost, (_, want_names) = _brute(t, y, {"steady": 0, "transient": 1}, penalty=0.5, subtypes=subtypes,
                                        events=events, **kw)
    got = optimal_segments(t, y, {"steady": 0, "transient": 1}, method="exact", penalty=0.5, subtypes=subtypes,
                           events=events, **kw)
    assert got.objective == pytest.approx(want_cost, rel=1e-9, abs=1e-9)
    for s in got.segments:
        assert _meets_conditions(subtypes[s.subtype], t[s.slice], y[s.slice], _within(events, s.slice))
    for name, cap in caps.items():
        assert (got.counts[name] <= cap).all()


def test_an_event_condition_reads_the_mask_whatever_the_signal():
    """Decided on #65: a flag on a non-finite sample counts, even in a segment with no finite sample."""
    t = np.arange(12.0)
    y = np.sin(t)
    y[4:8] = [np.nan, np.inf, np.nan, -np.inf]
    heater = np.zeros(12, dtype=bool)
    heater[5] = True
    subtypes = {"driven": rs.Subtype("transient", events={"heater": True}),
                "undriven": rs.Subtype("transient", events={"heater": False})}
    r = optimal_segments(t, y, {"transient": 1}, method="subset", candidates=[0, 4, 8, 12], penalty=0,
                         max_duration_s={"transient": 4.0}, subtypes=subtypes, events={"heater": heater})
    assert [s.subtype for s in r.segments] == ["undriven", "driven", "undriven"]

def test_a_heat_up_cap_of_one_merges_a_days_two_heat_ups():
    """A two-stage heat-up: the heater switches on at 08:00 (20 to 50) and again at 09:00 (50 to 80)."""
    idx = pd.date_range("2026-03-02", periods=24 * 60, freq="1min")
    hour = (idx - idx[0]).total_seconds().to_numpy() / 3600
    y = np.interp(hour, [0, 8, 8.5, 9, 9.5, 16, 17, 24], [20, 20, 50, 50, 80, 80, 20, 20])
    y = y + np.random.default_rng(60).normal(0, 0.3, len(y))
    events = {"heater_on": np.isin(hour, [8.0, 9.0]), "heater_off": hour == 16.0}
    subtypes = {"heat_up": rs.Subtype("transient", events={"heater_on": True}),
                "cool_down": rs.Subtype("transient", events={"heater_off": True}),
                "plateau": rs.Subtype("steady", events={"heater_on": False, "heater_off": False})}
    kw = dict(method="exact", grid_s=600, per="1D", penalty=100.0, subtypes=subtypes, events=events,
              transitions={"steady": ["transient"], "transient": ["steady"]})

    def shape(r):
        return [(idx[s.start].strftime("%H:%M"), s.subtype) for s in r.segments]

    free = optimal_segments(idx, y, {"steady": 0, "transient": 1}, **kw)
    assert shape(free) == [("00:00", "plateau"), ("08:00", "heat_up"), ("08:30", "plateau"), ("09:00", "heat_up"),
                           ("09:30", "plateau"), ("16:00", "cool_down"), ("17:00", "plateau")]
    capped = optimal_segments(idx, y, {"steady": 0, "transient": 1}, max_counts={"heat_up": 1}, **kw)
    assert shape(capped) == [("00:00", "plateau"), ("08:00", "heat_up"), ("09:30", "plateau"),
                             ("16:00", "cool_down"), ("17:00", "plateau")]
    assert capped.counts["heat_up"].tolist() == [1] and capped.objective > free.objective


@pytest.mark.parametrize("where", ["last of the rise", "first of the fall"])
def test_refine_never_pushes_an_event_across_a_boundary(where):
    """Without the subtypes, refine moves both transients' boundaries past these flagged samples."""
    t, y, (up, down) = _startup_record()
    cands = [0, up + 37, up + 300, down - 41, down + 150, len(y)]
    switch = np.zeros(len(y), dtype=bool)
    switch[up + 299 if where == "last of the rise" else down - 41] = True
    events = {"switch": switch}
    subtypes = {"driven": rs.Subtype("transient", events={"switch": True}),
                "undriven": rs.Subtype("transient", events={"switch": False})}
    kw = dict(candidates=cands, penalty=10.0, max_counts={"steady": 3, "transient": 2})
    models = {"steady": 0, "transient": 1}
    sub = optimal_segments(t, y, models, method="subset", subtypes=subtypes, events=events, **kw)
    driven = [i for i, s in enumerate(sub.segments) if s.subtype == "driven"]
    assert len(driven) == 1
    for refine_s in (30.0, 120.0, 400.0):
        ref = optimal_segments(t, y, models, method="refine", refine_s=refine_s, subtypes=subtypes, events=events, **kw)
        assert ref.objective <= sub.objective
        assert [s.subtype for s in ref.segments] == [s.subtype for s in sub.segments]
        for s in ref.segments:
            if s.subtype is not None:
                assert _meets_conditions(subtypes[s.subtype], t[s.slice], y[s.slice], _within(events, s.slice))
        free = optimal_segments(t, y, models, method="refine", refine_s=refine_s, **kw)
        assert not switch[free.segments[driven[0]].slice].any()


def test_event_validation():
    heat_up = rs.Subtype("transient", events={"heater": np.True_})
    assert heat_up.events == (("heater", True),) and rs.Subtype("transient", events={}).events is None
    assert heat_up == rs.Subtype("transient", events={"heater": True})
    both = [rs.Subtype("transient", events=dict(e)) for e in ([("a", True), ("b", False)], [("b", False), ("a", True)])]
    assert both[0] == both[1] and hash(both[0]) == hash(both[1]) and dict(both[0].events) == {"a": True, "b": False}
    assert pickle.loads(pickle.dumps(heat_up)) == heat_up  # plain data, for batch runs across processes
    with pytest.raises(ValueError, match=r"Subtype events must be a mapping \{mask name: True or False\}"):
        rs.Subtype("transient", events=["heater"])
    for name in ("", 3):
        with pytest.raises(ValueError, match="a Subtype events mask name must be a non-empty string"):
            rs.Subtype("transient", events={name: True})
    for bad in (1, "yes", None):
        with pytest.raises(ValueError, match=r"Subtype events\['heater'\] must be True or False"):
            rs.Subtype("transient", events={"heater": bad})

    t = np.arange(10.0)
    mask = np.zeros(10, dtype=bool)

    def run(events, subtypes=None):
        return optimal_segments(t, np.zeros(10), {"transient": 1}, method="exact", events=events,
                                subtypes={"heat_up": heat_up} if subtypes is None else subtypes)

    with pytest.raises(ValueError, match="subtype 'heat_up' needs event mask 'heater', which events does not include"):
        run(None)
    with pytest.raises(ValueError, match="subtype 'heat_up' needs event mask 'heater', which events does not include"):
        run({"valve": mask})
    with pytest.raises(ValueError, match=r"events must be a mapping \{name: boolean mask\}; give a single mask a name"):
        run(mask)
    with pytest.raises(ValueError, match="an event mask's name must be a non-empty string"):
        run({"heater": mask, 3: mask})
    for bad in (mask[:9], np.zeros((10, 1), dtype=bool)):
        with pytest.raises(ValueError, match=r"events\['heater'\] must be a one-dimensional mask with one value "
                                             r"per sample of t \(10\)"):
            run({"heater": bad})
    for bad in (mask.astype(int), mask.astype(float), [0] * 10):
        with pytest.raises(ValueError, match=r"events\['heater'\] must be boolean, got values of type"):
            run({"heater": bad})
    with pytest.raises(ValueError, match=r"events\['valve'\] must be boolean"):  # a mask no condition names
        run({"heater": mask, "valve": mask.astype(int)})
    r = run({"heater": list(mask), "valve": pd.Series(mask)}, subtypes={"heat_up": heat_up,
                                                                       "other": rs.Subtype("transient")})
    assert [s.subtype for s in r.segments] == ["other"] and list(r.info["events"]) == ["heater", "valve"]
    given = np.zeros(10, dtype=bool)
    r = run({"heater": given}, subtypes={"heat_up": heat_up, "other": rs.Subtype("transient")})
    given[3] = True  # the caller reuses its mask: the result keeps the mask it used
    assert not r.info["events"]["heater"].any()

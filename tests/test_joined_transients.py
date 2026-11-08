"""Joined transients: each transient is the line between its steady neighbours' levels (#179)."""

import math

import numpy as np
import pandas as pd
import pytest

import rocksteady as rs
from rocksteady.fit import fit_model
from rocksteady.optimal import optimal_segments
from rocksteady.states import Segment, State

ST, TR = State.STEADY, State.TRANSIENT
MODELS = {"steady": 0, "transient": 1}


# -- an independent reference: the joined fit from the samples, and brute force over segmentations ----------


def _level(y):
    v = y[np.isfinite(y)]
    return float(v.mean()) if len(v) else math.nan


def _joined_sse(t, y, bounds, states):
    """
    The SSE of a segmentation under the joined fit, from the samples: a steady segment's mean; a transient's line
    from the previous level at its first sample to the next level at the next segment's first sample; at the
    record's start or end, the least-squares line through the one joined point; the whole record, a free line.
    Infinite when a transient borders a transient or a steady segment without a finite sample.
    """
    total = 0.0
    for i, st in enumerate(states):
        p, q = bounds[i], bounds[i + 1]
        tt, yy = t[p:q], y[p:q]
        ok = np.isfinite(yy)
        if st == "steady":
            lv = _level(yy)
            total += float(((yy[ok] - lv) ** 2).sum()) if ok.any() else 0.0
            continue
        near = []
        for j in (i - 1, i + 1):
            if 0 <= j < len(states):
                if states[j] != "steady":
                    return math.inf
                lv = _level(y[bounds[j]:bounds[j + 1]])
                if math.isnan(lv):
                    return math.inf
                near.append(lv)
            else:
                near.append(None)
        before, after = near
        if before is None and after is None:
            r = fit_model(1, tt, yy).residuals
            total += float(np.nansum(r * r))
            continue
        if before is not None and after is not None:
            line = before + (after - before) * (tt - t[p]) / (t[q] - t[p])
        else:
            t0, lv = (t[p], before) if after is None else (t[q], after)
            d = tt[ok] - t0
            b = float(d @ (yy[ok] - lv) / (d @ d)) if d @ d > 0 else 0.0
            line = lv + b * (tt - t0)
        total += float(((yy[ok] - line[ok]) ** 2).sum())
    return total


def _brute_joined(t, y, *, cuts, penalty, choices, transitions=None, initial=None, final=None, caps=None, per=None,
                  overflow=None, min_d=None, accepts=None, free_param_penalty=0.0):
    """
    The best joined objective over every segmentation on ``cuts`` and every choice per segment, by exhaustive
    search; a branch is cut only when it breaks a constraint. ``choices`` maps a choice's name to its state;
    ``penalty`` maps a choice to its penalty, and a transient adds ``free_param_penalty`` for each free parameter
    of its joined line (one at the record's start or end, two for the whole record). ``caps`` counts per period of
    ``per`` seconds (one segment once against each period it touches), soft for a name in ``overflow``. A
    transient never follows a transient. Returns (objective, [(bounds, names)] within 1e-9 of it).
    """
    n = len(y)
    dt = float(np.median(np.diff(t)))
    pid = np.zeros(n, dtype=int) if per is None else np.floor(t / per).astype(int)
    caps, soft = dict(caps or {}), dict(overflow or {})
    found = []

    def search(p, path, counts, charge):
        if p == n:
            names = [c for _, _, c in path]
            if final is not None and names[-1] not in final:
                return
            bounds = [0] + [q for _, q, _ in path]
            states = [choices[c] for c in names]
            sse = _joined_sse(t, y, bounds, states)
            pen = 0.0
            for i, c in enumerate(names):
                pen += penalty[c]
                if choices[c] == "transient":
                    pen += free_param_penalty * ((i == 0) + (i == len(names) - 1))
            found.append((sse + pen + charge, bounds, tuple(names)))
            return
        prev = path[-1][2] if path else None
        for q in sorted(c for c in cuts if c > p) + ([n] if n not in cuts else []):
            for c, st in choices.items():
                if prev is None and initial is not None and c not in initial:
                    continue
                if prev is not None and choices[prev] == "transient" and st == "transient":
                    continue
                if prev is not None and transitions is not None and c not in transitions.get(prev, ()):
                    continue
                if t[q - 1] - t[p] + dt < (min_d or {}).get(c, 0.0) - 1e-9:
                    continue
                if accepts is not None and c in accepts and not accepts[c](y[p:q]):
                    continue
                new, extra, broken = dict(counts), 0.0, False
                for key in {c, st}:
                    if key not in caps:
                        continue
                    for pp in range(pid[p], pid[q - 1] + 1):
                        new[(pp, key)] = new.get((pp, key), 0) + 1
                        if new[(pp, key)] > caps[key]:
                            if key in soft:
                                extra += soft[key]
                            else:
                                broken = True
                if not broken:
                    search(q, path + [(p, q, c)], new, charge + extra)

    search(0, [], {}, 0.0)
    if not found:
        return math.inf, []
    best = min(v for v, _, _ in found)
    return best, [(b, c) for v, b, c in found if v <= best + 1e-9 * max(1.0, abs(best))]


def _names(result):
    return tuple(s.subtype or s.state.value for s in result.segments)


def _bounds(result):
    return [0] + [s.stop for s in result.segments]


def _step_record(seed, n=9):
    """A noisy step between two levels, with a short ramp: a record whose best fit has a transient."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    a = int(rng.integers(2, n - 4))
    clean = np.interp(t, [a, a + 2], [0.0, 4.0])
    return t, clean + rng.normal(0, 0.2, n)


# -- the program is optimal under the joined objective ---------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_exact_is_the_best_joined_segmentation(seed):
    t, y = _step_record(seed)
    pen = {"steady": 0.3, "transient": 0.2}
    res = optimal_segments(t, y, MODELS, method="exact", penalty=pen, join_transients=True)
    best, optima = _brute_joined(t, y, cuts=range(1, len(y)), penalty=pen,
                                 choices={"steady": "steady", "transient": "transient"})
    assert res.objective == pytest.approx(best, rel=1e-9, abs=1e-12)
    assert (_bounds(res), _names(res)) in optima
    assert res.sse == pytest.approx(_joined_sse(t, y, _bounds(res), [s.state.value for s in res.segments]),
                                    rel=1e-9, abs=1e-12)


SUBTYPES = {"low": rs.Subtype("steady", level=(None, 2.0)), "high": rs.Subtype("steady", level=(2.0, None)),
            "up": rs.Subtype("transient"), "down": rs.Subtype("transient")}
SUB_CHOICES = {"low": "steady", "high": "steady", "up": "transient", "down": "transient"}
SUB_ACCEPTS = {"low": lambda y: np.isfinite(y).any() and np.nanmean(y) < 2.0,
               "high": lambda y: np.isfinite(y).any() and np.nanmean(y) >= 2.0}
PLAIN = {"steady": "steady", "transient": "transient"}


def _case(case):
    """``(optimal_segments keywords, _brute_joined keywords)`` for one set of constraints."""
    pen = {"steady": 0.3, "transient": 0.2}
    if case == "subtypes":
        pen = {"low": 0.3, "high": 0.4, "up": 0.2, "down": 0.25}
        trans = {"low": ["up"], "up": ["high"], "high": ["down"], "down": ["low"]}
        return (dict(subtypes=SUBTYPES, transitions=trans, initial={"low"}, penalty=pen),
                dict(choices=SUB_CHOICES, accepts=SUB_ACCEPTS, transitions=trans, initial={"low"}, penalty=pen))
    if case == "caps":
        caps = {"steady": 2, "transient": 1}
        return (dict(per=4.0, max_counts=caps, penalty=0.0),
                dict(choices=PLAIN, per=4.0, caps=caps, penalty={"steady": 0.0, "transient": 0.0}))
    if case == "soft caps":
        caps, soft = {"steady": 1, "transient": 0}, {"steady": 0.15, "transient": 0.4}
        return (dict(per=4.0, max_counts=caps, overflow_penalty=soft, penalty=pen),
                dict(choices=PLAIN, per=4.0, caps=caps, overflow=soft, penalty=pen))
    if case == "first and last":
        return (dict(initial={"transient"}, final={"steady"}, penalty=pen),
                dict(choices=PLAIN, initial={"transient"}, final={"steady"}, penalty=pen))
    if case == "last transient":
        return (dict(initial={"steady"}, final={"transient"}, penalty=pen),
                dict(choices=PLAIN, initial={"steady"}, final={"transient"}, penalty=pen))
    assert case == "durations"
    return (dict(min_duration_s={"steady": 3.0}, penalty=pen),
            dict(choices=PLAIN, min_d={"steady": 3.0}, penalty=pen))


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("case", ["subtypes", "caps", "soft caps", "first and last", "last transient", "durations"])
def test_exact_is_the_best_joined_segmentation_under_constraints(seed, case):
    t, y = _step_record(seed)
    kw, ref = _case(case)
    res = optimal_segments(t, y, MODELS, method="exact", join_transients=True, **kw)
    best, optima = _brute_joined(t, y, cuts=range(1, len(y)), **ref)
    assert res.objective == pytest.approx(best, rel=1e-9, abs=1e-12)
    assert (_bounds(res), _names(res)) in optima


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("case", ["subtypes", "caps", "soft caps"])
def test_subset_is_the_best_joined_segmentation_on_its_candidates(seed, case):
    t, y = _step_record(seed)
    kw, ref = _case(case)
    cuts = [2, 3, 5, 6, 8]
    res = optimal_segments(t, y, MODELS, method="subset", candidates=cuts, join_transients=True, **kw)
    best, optima = _brute_joined(t, y, cuts=cuts, **ref)
    assert res.objective == pytest.approx(best, rel=1e-9, abs=1e-12)
    assert (_bounds(res), _names(res)) in optima
    assert set(_bounds(res)) <= {0, len(y), *cuts}


@pytest.mark.parametrize("initial, final, states, free", [
    ({"transient"}, {"steady"}, [TR, ST, TR, ST], 1),  # the first transient is joined on its inner side only
    ({"steady"}, {"transient"}, [ST, TR, ST, TR], 1),  # the last one
    ({"transient"}, {"transient"}, [TR, ST, TR, ST, TR], 2),  # both
])
def test_bic_counts_a_joined_transients_free_parameters(initial, final, states, free):
    # sigma^2 (p + 1) ln n: a steady constant has p = 1, a transient joined on both sides p = 0, so the steady
    # penalty is twice the transient's; each free parameter of a transient at an end adds the transient's again.
    t = np.arange(48, dtype=float)
    rng = np.random.default_rng(3)
    y = np.interp(t, [0, 6, 20, 24, 40, 48], [3.0, 0.0, 0.0, 5.0, 5.0, 1.0]) + rng.normal(0, 0.05, 48)
    if final == {"steady"}:
        y[40:] = 5.0 + rng.normal(0, 0.05, 8)
    if initial == {"steady"}:
        y[:6] = rng.normal(0, 0.05, 6)
    res = optimal_segments(t, y, MODELS, method="exact", grid_s=2.0, initial=initial, final=final,
                           join_transients=True)
    assert [s.state for s in res.segments] == states
    pen = res.penalty
    assert pen["steady"] == pytest.approx(2 * pen["transient"], rel=1e-12)
    n_steady, n_trans = states.count(ST), states.count(TR)
    expected = n_steady * pen["steady"] + (n_trans + free) * pen["transient"]
    assert res.objective - res.sse == pytest.approx(expected, rel=1e-9)
    unjoined = optimal_segments(t, y, MODELS, method="exact", grid_s=2.0, initial=initial, final=final)
    assert unjoined.penalty["transient"] == pytest.approx(1.5 * unjoined.penalty["steady"], rel=1e-12)


def test_a_transient_that_is_the_whole_record_is_a_free_line_with_two_free_parameters():
    t = np.arange(30, dtype=float)
    y = 0.2 * t + np.random.default_rng(6).normal(0, 0.05, 30)
    res = optimal_segments(t, y, MODELS, method="subset", candidates=[], initial={"transient"}, join_transients=True)
    assert [(s.start, s.stop, s.state) for s in res.segments] == [(0, 30, TR)]
    free = fit_model(1, t, y).residuals
    assert res.sse == pytest.approx(float(free @ free), rel=1e-9)
    assert res.objective - res.sse == pytest.approx(3 * res.penalty["transient"], rel=1e-9)


# -- what the option refuses -------------------------------------------------------------------------------------


@pytest.mark.parametrize("models, named", [
    ({"steady": 0, "transient": "exp"}, "state 'transient'"),
    ({"steady": 1, "transient": 1}, "state 'steady'"),
    ({"steady": {"model": "poly", "order": 0}, "transient": {"model": "poly", "order": 2}}, "state 'transient'"),
])
def test_models_other_than_a_constant_and_a_line_are_refused(models, named):
    t, y = _step_record(0)
    with pytest.raises(ValueError, match=named):
        optimal_segments(t, y, models, method="exact", join_transients=True)


def test_a_subtypes_own_model_that_is_not_a_line_is_refused():
    # The offending subtype sits between two that are fine, so neither the first nor the last is all that is checked.
    t, y = _step_record(0)
    subtypes = {"a": rs.Subtype("transient", model=1), "b": rs.Subtype("transient", model="first_order"),
                "c": rs.Subtype("transient", model={"model": "poly", "order": 1})}
    with pytest.raises(ValueError, match="subtype 'b'"):
        optimal_segments(t, y, MODELS, method="exact", subtypes=subtypes, join_transients=True)
    del subtypes["b"]
    optimal_segments(t, y, MODELS, method="exact", subtypes=subtypes, join_transients=True)


def test_indeterminate_is_refused():
    t, y = _step_record(0)
    with pytest.raises(ValueError, match="'indeterminate'"):
        optimal_segments(t, y, {**MODELS, "indeterminate": 1}, method="exact", join_transients=True)


@pytest.mark.parametrize("value", [1, 0, "yes", None])
def test_join_transients_must_be_a_boolean(value):
    t, y = _step_record(0)
    with pytest.raises(ValueError, match="join_transients must be True or False"):
        optimal_segments(t, y, MODELS, method="exact", join_transients=value)


def test_a_numpy_boolean_is_read_as_the_boolean():
    t, y = _step_record(1)
    pen = {"steady": 0.3, "transient": 0.2}
    a = optimal_segments(t, y, MODELS, method="exact", penalty=pen, join_transients=np.True_)
    b = optimal_segments(t, y, MODELS, method="exact", penalty=pen, join_transients=True)
    off = optimal_segments(t, y, MODELS, method="exact", penalty=pen, join_transients=np.False_)
    assert (a.segments, a.objective) == (b.segments, b.objective)
    assert off.objective != a.objective
    assert type(a.models["join_transients"]) is bool
    assert "join_transients" not in off.models


# -- the joined fit ---------------------------------------------------------------------------------------------


def _adjacent_transients(segments):
    return [i for i in range(1, len(segments)) if segments[i - 1].state == TR and segments[i].state == TR]


@pytest.mark.parametrize("transitions", [None, {"steady": ["steady", "transient"],
                                                 "transient": ["transient", "steady"]}])
def test_no_transient_follows_a_transient(transitions):
    # A kinked ramp: without the option, two lines fit it best, one after the other.
    t = np.arange(30, dtype=float)
    rng = np.random.default_rng(5)
    y = np.interp(t, [0, 5, 10, 15, 30], [0.0, 0.0, 2.0, 8.0, 8.0]) + rng.normal(0, 0.02, 30)
    kw = dict(method="exact", transitions=transitions, penalty={"steady": 0.01, "transient": 0.01})
    unjoined = optimal_segments(t, y, MODELS, **kw)
    assert _adjacent_transients(unjoined.segments)
    res = optimal_segments(t, y, MODELS, join_transients=True, **kw)
    assert not _adjacent_transients(res.segments)
    assert any(s.state == TR for s in res.segments)


def test_steady_segments_still_meet_in_a_step():
    # Steady -> steady stays allowed: a sharp step between two levels is two steady segments, joined by nothing.
    t = np.arange(20, dtype=float)
    y = np.where(t < 10, 1.0, 4.0) + np.random.default_rng(2).normal(0, 0.05, 20)
    res = optimal_segments(t, y, MODELS, method="exact", penalty={"steady": 0.1, "transient": 0.1},
                           join_transients=True)
    assert [(s.start, s.stop, s.state) for s in res.segments] == [(0, 10, ST), (10, 20, ST)]


def _report(t, y, segments, spec):
    return rs.report.fit_report(pd.DataFrame({"y": y}), segments, {"y": spec}, t=t)


def _two_steps(seed=4, n=60):
    """Up a ramp to a plateau, then down a ramp to a lower one, with noise."""
    t = np.arange(n, dtype=float)
    rng = np.random.default_rng(seed)
    clean = np.interp(t, [0, 10, 18, 35, 41, 60], [1.0, 1.0, 6.0, 6.0, 3.0, 3.0])
    return t, clean + rng.normal(0, 0.05, n)


def test_each_transient_meets_its_neighbours_levels():
    t, y = _two_steps()
    res = optimal_segments(t, y, MODELS, method="exact", grid_s=2.0, join_transients=True)
    segs = res.segments
    assert [s.state for s in segs] == [ST, TR, ST, TR, ST]
    rep = _report(t, y, segs, res.models)
    fits = rep.blocks[0].fit_results
    for i, s in enumerate(segs):
        if s.state != TR:
            continue
        before, after = segs[i - 1], segs[i + 1]
        # At its first sample, the previous segment's level; at the next segment's first sample, that one's.
        assert fits[i].predict(np.array([t[s.start]]))[0] == pytest.approx(_level(y[before.slice]), rel=1e-12)
        assert fits[i].predict(np.array([t[after.start]]))[0] == pytest.approx(_level(y[after.slice]), rel=1e-12)
        assert rep.blocks[0].fit.fitted[s.start] == pytest.approx(_level(y[before.slice]), rel=1e-12)


def test_the_report_fits_as_the_optimiser_did():
    t, y = _two_steps()
    res = optimal_segments(t, y, MODELS, method="exact", grid_s=2.0, join_transients=True)
    rep = _report(t, y, res.segments, res.models)
    r = rep.blocks[0].fit.residuals
    assert float(np.nansum(r * r)) == pytest.approx(res.sse, rel=1e-9)
    unjoined = _report(t, y, res.segments, MODELS)
    r_free = unjoined.blocks[0].fit.residuals
    assert float(np.nansum(r_free * r_free)) < res.sse  # free lines fit the same segments better
    # The report says the transients were joined; the unjoined report does not.
    assert rep.blocks[0].title.endswith("; transients joined)")
    assert rep.provenance["models"]["y"]["join_transients"] is True
    assert "joined" not in unjoined.blocks[0].title
    assert "join_transients" not in unjoined.provenance["models"]["y"]


JOINED = {**MODELS, "join_transients": True}


@pytest.mark.parametrize("states, named", [
    ([ST, TR, TR, ST], r"segment after it \(20:30\) is transient"),
    ([ST, TR, State.INDETERMINATE, ST], r"segment after it \(20:30\) is indeterminate"),
    ([ST, State.INDETERMINATE, TR, ST], r"segment before it \(10:20\) is indeterminate"),
])
def test_the_report_refuses_a_transient_beside_a_segment_that_is_not_steady(states, named):
    t, y = _two_steps(n=40)
    bounds = [0, 10, 20, 30, 40]
    segs = [Segment(bounds[i], bounds[i + 1], st) for i, st in enumerate(states)]
    with pytest.raises(ValueError, match=named):
        _report(t, y, segs, JOINED)
    _report(t, y, segs, MODELS)  # unjoined, the same segments are fitted


@pytest.mark.parametrize("empty, named", [(slice(20, 30), r"after it \(20:30\)"),
                                          (slice(0, 10), r"before it \(0:10\)")])
def test_the_report_refuses_a_transient_beside_a_steady_segment_without_a_sample(empty, named):
    t, y = _two_steps(n=40)
    y[empty] = np.nan
    segs = [Segment(0, 10, ST), Segment(10, 20, TR), Segment(20, 30, ST), Segment(30, 40, ST)]
    with pytest.raises(ValueError, match=named + " has no finite sample"):
        _report(t, y, segs, JOINED)


@pytest.mark.parametrize("value", [1, "yes", None])
def test_the_specs_key_must_be_a_boolean(value):
    t, y = _two_steps(n=40)
    with pytest.raises(ValueError, match="model for 'y': a model spec's 'join_transients' must be True or False"):
        _report(t, y, [Segment(0, 40, ST)], {**MODELS, "join_transients": value})


def test_the_report_refuses_a_joined_spec_that_is_not_a_constant_and_a_line():
    t, y = _two_steps(n=40)
    with pytest.raises(ValueError, match="state 'transient' has model 'exp'"):
        _report(t, y, [Segment(0, 40, ST)], {"steady": 0, "transient": "exp", "join_transients": True})
    with pytest.raises(ValueError, match="transient 10:20 has model"):
        segs = [Segment(0, 10, ST), Segment(10, 20, TR, subtype="pulse"), Segment(20, 40, ST)]
        _report(t, y, segs, {**JOINED, "subtypes": {"pulse": 0}})  # a constant, on a transient
    with pytest.raises(ValueError, match="steady segment 20:40 has model"):
        segs = [Segment(0, 10, ST), Segment(10, 20, TR), Segment(20, 40, ST, subtype="hot")]
        _report(t, y, segs, {**JOINED, "subtypes": {"hot": 1}})  # a line, on a steady segment


def test_the_specs_key_takes_a_numpy_boolean():
    t, y = _two_steps(n=40)
    segs = [Segment(0, 10, ST), Segment(10, 20, TR), Segment(20, 40, ST)]
    a = _report(t, y, segs, {**MODELS, "join_transients": np.True_}).blocks[0].fit.fitted
    b = _report(t, y, segs, JOINED).blocks[0].fit.fitted
    c = _report(t, y, segs, {**MODELS, "join_transients": np.False_}).blocks[0].fit.fitted
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def _ramp_then_nothing():
    """A plateau, a ramp, then no finite sample."""
    t = np.arange(20, dtype=float)
    y = np.where(t < 5, 0.0, np.where(t < 10, t - 5.0, np.nan)) + np.random.default_rng(1).normal(0, 0.01, 20)
    return t, y


def test_the_optimiser_does_not_join_a_transient_to_a_steady_segment_without_a_sample():
    t, y = _ramp_then_nothing()
    kw = dict(method="subset", candidates=[5, 10], final={"steady"}, penalty={"steady": 0.05, "transient": 0.2})
    unjoined = optimal_segments(t, y, MODELS, **kw)
    assert [(s.start, s.stop, s.state) for s in unjoined.segments] == [(0, 5, ST), (5, 10, TR), (10, 20, ST)]
    res = optimal_segments(t, y, MODELS, join_transients=True, **kw)
    segs = res.segments
    for i, s in enumerate(segs):
        if s.state == TR:
            for near in segs[max(0, i - 1):i] + segs[i + 1:i + 2]:
                assert np.isfinite(y[near.slice]).sum() > 0
    # The best joined segmentation on these candidates, where a transient beside the empty segment costs infinity.
    best, optima = _brute_joined(t, y, cuts=[5, 10], penalty={"steady": 0.05, "transient": 0.2}, choices=PLAIN,
                                 final={"steady"})
    assert res.objective == pytest.approx(best, rel=1e-9)
    assert (_bounds(res), _names(res)) in optima


def test_with_no_other_segmentation_the_empty_neighbour_is_infeasible():
    t, y = _ramp_then_nothing()
    with pytest.raises(ValueError, match=r"none can continue from 10\.0 s.*cannot border a steady segment that has "
                                         "no finite sample"):
        optimal_segments(t, y, MODELS, method="subset", candidates=[10], initial={"transient"}, final={"steady"},
                         penalty=0.1, join_transients=True)


# -- the caller's day: a reactor's power with steady constants and linear transients ----------------------------

RATED = 900e3
NATIVE = dict(
    per="1D",
    subtypes={"zero_power": rs.Subtype("steady", level=(None, 1.0)),
              "at_power": rs.Subtype("steady", level=(1.0, None)),
              "startup": rs.Subtype("transient"), "shutdown": rs.Subtype("transient")},
    max_counts={"zero_power": 2, "startup": 1, "at_power": 1, "shutdown": 1},
    transitions={"zero_power": ["startup"], "startup": ["at_power"], "at_power": ["shutdown"],
                 "shutdown": ["zero_power"]},
    initial={"steady"}, final={"steady"},
)


def _day(freq, knots, seed=0):
    """A day of power through ``knots`` ``[(hh:mm, watts)]``, joined by lines, with 0.3 % noise and 0.01 W more."""
    idx = pd.date_range("2025-05-07 00:00:00", "2025-05-07 23:59:59", freq=freq)
    minute = (idx - idx[0]).total_seconds().to_numpy() / 60
    at = [int(h) * 60 + int(m) for h, m in (k.split(":") for k, _ in knots)]
    clean = np.interp(minute, at, [w for _, w in knots])
    rng = np.random.default_rng(seed)
    return idx, clean + rng.normal(size=len(idx)) * (0.003 * clean + 0.01)


def _candidates(idx, spacing_s=300.0):
    """The first sample of each ``spacing_s`` bin, counted from the first sample."""
    bins = np.floor((idx - idx[0]).total_seconds().to_numpy() / spacing_s)
    return np.flatnonzero(np.diff(bins, prepend=bins[0] - 1.0))


SCRAM_DAY = [("00:00", 0.0), ("08:45", 0.0), ("09:00", RATED), ("16:00", RATED), ("16:01", 0.0), ("16:30", 0.0),
             ("16:38", RATED), ("17:50", RATED), ("18:00", 0.0), ("23:59", 0.0)]


def _flat_runs_s(idx, y, segs, sigma):
    """The longest run of samples, in seconds, that any transient holds within 3 sigma of a neighbour's level."""
    longest = 0.0
    times = (idx - idx[0]).total_seconds().to_numpy()
    for i, s in enumerate(segs):
        if s.state != TR:
            continue
        for near in segs[max(0, i - 1):i] + segs[i + 1:i + 2]:
            close = np.abs(y[s.slice] - _level(y[near.slice])) <= 3 * sigma
            run_start = None
            for k, c in enumerate(close):
                if c and run_start is None:
                    run_start = k
                if run_start is not None and (not c or k == len(close) - 1):
                    end = k if c else k - 1
                    longest = max(longest, times[s.start + end] - times[s.start + run_start])
                    run_start = None
    return longest


@pytest.mark.parametrize("freq, method", [("1min", "exact"), ("1s", "refine")])
def test_a_transient_does_not_take_a_plateau_that_a_cap_keeps_from_being_steady(freq, method):
    # A scram at 16:00 and a restart at 16:30, under at most one at-power segment a day: without the option,
    # a flat startup takes the morning plateau and the one at-power segment takes the dip.
    idx, y = _day(freq, SCRAM_DAY)
    window = 300.0  # the grid's spacing for exact, how far refine moves a boundary
    kw = dict(method=method, **NATIVE)
    kw.update(dict(grid_s=window) if method == "exact" else dict(candidates=_candidates(idx), refine_s=window))
    unjoined = optimal_segments(idx, y, MODELS, **kw)
    assert [s.subtype for s in unjoined.segments] == ["zero_power", "startup", "at_power", "shutdown", "zero_power"]
    assert idx[unjoined.segments[1].stop] > pd.Timestamp("2025-05-07 15:00")  # the flat startup
    dip = unjoined.segments[2]  # the one at-power segment, on the dip
    assert pd.Timestamp("2025-05-07 15:55") <= idx[dip.start] and idx[dip.stop - 1] <= pd.Timestamp("2025-05-07 16:45")
    res = optimal_segments(idx, y, MODELS, join_transients=True, **kw)
    names = [s.subtype for s in res.segments]
    assert names == ["zero_power", "startup", "at_power", "shutdown", "zero_power"]
    startup_end = idx[res.segments[1].stop]
    assert abs((startup_end - pd.Timestamp("2025-05-07 09:00")).total_seconds()) <= window
    assert _flat_runs_s(idx, y, res.segments, sigma=0.003 * RATED) <= window


GAP_DAY = [("00:00", 0.0), ("08:15", 0.0), ("08:30", RATED), ("16:30", RATED), ("16:45", 0.0), ("23:59", 0.0)]


@pytest.mark.parametrize("rows", ["dropped", "nan"])
@pytest.mark.parametrize("gap, which", [(("08:14", "08:31"), "startup"), (("16:25", "16:50"), "shutdown")])
def test_a_gap_over_a_transient_leaves_it_no_finite_sample_but_the_last_before_the_gap(rows, gap, which):
    # With the gap's rows missing, the transient holds exactly that sample; with them there as NaN, at most it.
    idx, y = _day("1min", GAP_DAY)
    missing = (idx >= pd.Timestamp(f"2025-05-07 {gap[0]}")) & (idx <= pd.Timestamp(f"2025-05-07 {gap[1]}"))
    if rows == "dropped":
        idx, y = idx[~missing], y[~missing]
    else:
        y = np.where(missing, np.nan, y)
    finite = np.flatnonzero(np.isfinite(y))
    before = finite[idx[finite] < pd.Timestamp(f"2025-05-07 {gap[0]}")][-1]
    after = finite[idx[finite] > pd.Timestamp(f"2025-05-07 {gap[1]}")][0]
    res = optimal_segments(idx, y, MODELS, method="refine", candidates=_candidates(idx), refine_s=300.0,
                           join_transients=True, **NATIVE)
    assert [s.subtype for s in res.segments] == ["zero_power", "startup", "at_power", "shutdown", "zero_power"]
    i = [s.subtype for s in res.segments].index(which)
    tr, nxt = res.segments[i], res.segments[i + 1]
    held = list(np.flatnonzero(np.isfinite(y[tr.slice])) + tr.start)
    assert held == [before] if rows == "dropped" else set(held) <= {before}
    next_finite = np.flatnonzero(np.isfinite(y[nxt.slice])) + nxt.start
    assert next_finite[0] == after
    if rows == "dropped":
        assert nxt.start == after


@pytest.mark.parametrize("seed", range(4))
def test_refine_moves_boundaries_under_the_joined_fit(seed):
    t, y = _two_steps(seed=seed)
    pen = {"steady": 0.3, "transient": 0.2}
    cuts = list(range(0, 60, 6))
    subset = optimal_segments(t, y, MODELS, method="subset", candidates=cuts, penalty=pen, join_transients=True)
    res = optimal_segments(t, y, MODELS, method="refine", candidates=cuts, refine_s=3.0, penalty=pen,
                           join_transients=True)
    assert res.objective <= subset.objective
    assert _bounds(res) != _bounds(subset)  # off the candidates: the ramps start and end between them
    states = [s.state.value for s in res.segments]
    assert res.sse == pytest.approx(_joined_sse(t, y, _bounds(res), states), rel=1e-9)
    assert res.objective == pytest.approx(res.sse + sum(pen[st] for st in states), rel=1e-9)


def test_refine_scores_moves_by_the_joined_fit():
    # A curved rise: on this record the free line fits it best over a different span from the line joined to the
    # two levels, so moves scored by the free lines would stop elsewhere, at a worse joined objective.
    t = np.arange(60, dtype=float)
    rng = np.random.default_rng(7)
    y = np.where(t < 20, 1.0, 1.0 + 5.0 * (1 - np.exp(-(t - 20) / 4.0))) + rng.normal(0, 0.01, 60)
    pen = {"steady": 0.3, "transient": 0.2}
    kw = dict(candidates=[22, 30], penalty=pen, initial={"steady"}, final={"steady"},
              transitions={"steady": ["transient"], "transient": ["steady"]})
    subset = optimal_segments(t, y, MODELS, method="subset", join_transients=True, **kw)
    res = optimal_segments(t, y, MODELS, method="refine", refine_s=4.0, join_transients=True, **kw)
    unjoined = optimal_segments(t, y, MODELS, method="refine", refine_s=4.0, **kw)
    states = ["steady", "transient", "steady"]
    assert [s.state.value for s in res.segments] == states
    assert _bounds(subset) == [0, 22, 30, 60]
    assert _bounds(res) != _bounds(unjoined)
    assert res.sse == pytest.approx(_joined_sse(t, y, _bounds(res), states), rel=1e-9)
    assert res.sse < _joined_sse(t, y, _bounds(unjoined), states) < _joined_sse(t, y, _bounds(subset), states)


@pytest.mark.parametrize("own", [{"pulse": "exp"}, {"a": 1, "pulse": 2, "b": 0}])
def test_a_joined_spec_refuses_a_subtypes_model_that_is_not_a_constant_or_a_line(own):
    # Refused up front, whether or not a segment carries the subtype; in the second spec the offending one is
    # neither first nor last.
    t, y = _two_steps(n=40)
    carried = [Segment(0, 10, ST), Segment(10, 20, TR, subtype="pulse"), Segment(20, 40, ST)]
    for segs in ([Segment(0, 40, ST)], carried):
        with pytest.raises(ValueError, match="subtype 'pulse' has model"):
            _report(t, y, segs, {**JOINED, "subtypes": own})
    rs.fit.validate_models({"y": {**MODELS, "subtypes": own}})  # unjoined, the spec is fine


def test_the_report_joins_a_transient_at_each_end_on_its_inner_side():
    # Down from 3 to a plateau, up to another, down again: a transient starts and one ends the record.
    t = np.arange(48, dtype=float)
    rng = np.random.default_rng(3)
    y = np.interp(t, [0, 6, 20, 24, 40, 48], [3.0, 0.0, 0.0, 5.0, 5.0, 1.0]) + rng.normal(0, 0.05, 48)
    res = optimal_segments(t, y, MODELS, method="exact", grid_s=2.0, initial={"transient"}, final={"transient"},
                           join_transients=True)
    segs = res.segments
    assert [s.state for s in segs] == [TR, ST, TR, ST, TR]
    rep = _report(t, y, segs, res.models)
    r = rep.blocks[0].fit.residuals
    assert float(np.nansum(r * r)) == pytest.approx(res.sse, rel=1e-9)
    fits = rep.blocks[0].fit_results
    first, last = segs[0], segs[-1]
    # Inner sides: the first meets the next level at the next segment's first sample, the last starts from
    # the previous level at its own first sample.
    assert fits[0].predict(np.array([t[segs[1].start]]))[0] == pytest.approx(_level(y[segs[1].slice]), rel=1e-12)
    assert fits[-1].predict(np.array([t[last.start]]))[0] == pytest.approx(_level(y[segs[-2].slice]), rel=1e-12)
    # Outer sides: the least-squares slope through the joined point.
    for f, s, (t0, lv) in ((fits[0], first, (t[segs[1].start], _level(y[segs[1].slice]))),
                           (fits[-1], last, (t[last.start], _level(y[segs[-2].slice])))):
        d = t[s.slice] - t0
        slope = float(d @ (y[s.slice] - lv) / (d @ d))
        assert f.params["slope"] == pytest.approx(slope, rel=1e-9)
        assert f.n_params == 1
    assert fits[2].n_params == 0
    assert all("joined" in fits[i].flags for i in (0, 2, 4))


def test_the_report_fits_a_subtyped_result_as_the_optimiser_did():
    idx, y = _day("1min", SCRAM_DAY)
    res = optimal_segments(idx, y, MODELS, method="exact", grid_s=300.0, join_transients=True, **NATIVE)
    rep = rs.report.fit_report(pd.DataFrame({"y": y}, index=idx), res.segments, {"y": res.models})
    r = rep.blocks[0].fit.residuals
    assert float(np.nansum(r * r)) == pytest.approx(res.sse, rel=1e-9)
    assert list(rep.blocks[0].by_interval["subtype"]) == [s.subtype for s in res.segments]


def _overflow_cost(t, segs, per, caps, soft):
    """The soft caps' cost of a segmentation: each segment beyond a cap in a period, counted once per period."""
    pid = np.floor(t / per).astype(int)
    counts = {}
    for s in segs:
        for name in {s.state.value, s.subtype} - {None}:
            for p in range(pid[s.start], pid[s.stop - 1] + 1):
                counts[(p, name)] = counts.get((p, name), 0) + 1
    return sum(max(0, c - caps[name]) * soft[name] for (p, name), c in counts.items() if name in soft)


@pytest.mark.parametrize("seed", range(3))
def test_refine_keeps_soft_caps_under_the_joined_fit(seed):
    t, y = _two_steps(seed=seed)
    caps, soft = {"steady": 1, "transient": 0}, {"steady": 0.15, "transient": 0.4}
    pen = {"steady": 0.3, "transient": 0.2}
    kw = dict(per=20.0, max_counts=caps, overflow_penalty=soft, penalty=pen, candidates=list(range(0, 60, 6)))
    subset = optimal_segments(t, y, MODELS, method="subset", join_transients=True, **kw)
    res = optimal_segments(t, y, MODELS, method="refine", refine_s=3.0, join_transients=True, **kw)
    over = _overflow_cost(t, res.segments, 20.0, caps, soft)
    assert over > 0
    assert res.info["overflow_cost"] == pytest.approx(over, rel=1e-12)
    states = [s.state.value for s in res.segments]
    assert res.sse == pytest.approx(_joined_sse(t, y, _bounds(res), states), rel=1e-9)
    assert res.objective == pytest.approx(res.sse + sum(pen[st] for st in states) + over, rel=1e-9)
    assert res.objective <= subset.objective


# -- small records on which a careless search goes wrong -------------------------------------------------------


def _obeys(result, transitions, initial, final):
    names = _names(result)
    pairs = all(b in transitions[a] for a, b in zip(names, names[1:]))
    return pairs and names[0] in initial and names[-1] in final


@pytest.mark.parametrize("method", ["exact", "subset", "refine"])
def test_every_method_keeps_the_transitions_first_and_last_segments(method):
    # Equal penalties and a record whose best unconstrained fit breaks the sequence.
    t = np.arange(8, dtype=float)
    y = np.array([2.0, 5.0, 4.0, 3.0, 5.0, 5.0, 5.0, 5.0])
    pen = {"low": 0.3, "high": 0.3, "up": 0.3, "down": 0.3}
    trans = {"low": ["up"], "up": ["high"], "high": ["down"], "down": ["low"]}
    kw = dict(subtypes=SUBTYPES, transitions=trans, penalty=pen, initial={"low", "high"}, final={"up", "high"})
    if method != "exact":
        kw.update(candidates=[3, 4, 6])
    if method == "refine":
        kw.update(refine_s=2.0)
    res = optimal_segments(t, y, MODELS, method=method, join_transients=True, **kw)
    assert _obeys(res, trans, {"low", "high"}, {"up", "high"})
    if method != "refine":
        cuts = range(1, 8) if method == "exact" else [3, 4, 6]
        best, optima = _brute_joined(t, y, cuts=cuts, choices=SUB_CHOICES, accepts=SUB_ACCEPTS, transitions=trans,
                                     initial={"low", "high"}, final={"up", "high"}, penalty=pen)
        assert res.objective == pytest.approx(best, rel=1e-9)
        assert (_bounds(res), _names(res)) in optima


@pytest.mark.parametrize("y, cuts", [
    ([4.0, 3.0, 2.0, 3.5, 5.0, 5.0], [2, 3, 4]),  # a transient could end the record
    ([0.0, 1.7, 2.3, 0.8, 2.8, 2.4, 1.5, 3.0], [1, 2, 5]),  # or start it
    ([0.0, 4.0, 2.0, 1.0, 3.0, 0.0, 1.0], [1, 4, 5]),  # or be all of it
])
def test_bic_charges_a_transients_free_parameters_when_choosing(y, cuts):
    # On these records a transient at an end, or the whole record as one, would win if its free parameters were free.
    y = np.array(y)
    t = np.arange(len(y), dtype=float)
    res = optimal_segments(t, y, MODELS, method="subset", candidates=cuts, penalty="bic", join_transients=True)
    pen = res.penalty
    best, optima = _brute_joined(t, y, cuts=cuts, choices=PLAIN, penalty={"steady": pen["steady"],
                                 "transient": pen["transient"]}, free_param_penalty=pen["transient"])
    assert res.objective == pytest.approx(best, rel=1e-9)
    assert (_bounds(res), _names(res)) in optima


def test_a_soft_cap_of_zero_charges_the_first_segment_for_each_period_it_touches():
    t = np.arange(8, dtype=float)
    y = np.array([-0.4, -2.3, -2.4, -3.2, -2.1, -2.4, -2.3, -3.1])
    caps, soft = {"steady": 1, "transient": 0}, {"steady": 0.2, "transient": 0.3}
    res = optimal_segments(t, y, MODELS, method="subset", candidates=[1, 3, 7], per=4.0, max_counts=caps,
                           overflow_penalty=soft, penalty=0.1, join_transients=True)
    best, optima = _brute_joined(t, y, cuts=[1, 3, 7], choices=PLAIN, per=4.0, caps=caps, overflow=soft,
                                 penalty={"steady": 0.1, "transient": 0.1})
    assert res.objective == pytest.approx(best, rel=1e-9)
    assert (_bounds(res), _names(res)) in optima


@pytest.mark.parametrize("y, kw, expected, alternative", [
    # Moving the boundary at 3 changes the level the first transient is joined to.
    ([0.0, 0.0, 0.0, 2.5, 5.0, 1.0, 2.0, 2.0],
     dict(candidates=[1, 4, 5], refine_s=1.0, initial={"transient"}, final={"steady"}, penalty=0.2),
     [0, 2, 3, 5, 8], [0, 2, 4, 5, 8]),
    # Moving the boundary at 3 changes the level the last transient is joined to.
    ([3.0, 3.0, 3.0, 2.0, 1.0, 4.0, 4.0],
     dict(candidates=[1, 2, 4, 5], refine_s=2.0, subtypes=SUBTYPES, initial={"high"}, final={"low", "up"},
          transitions={"low": ["up"], "up": ["high"], "high": ["down"], "down": ["low"]},
          penalty={"low": 0.3, "high": 0.3, "up": 0.05, "down": 0.3}),
     [0, 2, 3, 5, 7], [0, 2, 4, 5, 7]),
])
def test_refine_costs_the_transient_beyond_the_moved_segment(y, kw, expected, alternative):
    # On these records the result has a lower joined objective than the alternative, where the boundary at 3
    # sits one sample later.
    y = np.array(y)
    t = np.arange(len(y), dtype=float)
    res = optimal_segments(t, y, MODELS, method="refine", join_transients=True, **kw)
    states = [s.state.value for s in res.segments]
    assert _bounds(res) == expected
    assert _joined_sse(t, y, expected, states) < _joined_sse(t, y, alternative, states)


def test_the_error_names_where_a_transient_is_stranded():
    # The only segmentation is steady, transient, then a steady segment without a sample (no steady segment may
    # last 10 s): the transient reaches 10 s, and nothing can follow it there.
    t, y = _ramp_then_nothing()
    t, y = t[:15], y[:15]
    with pytest.raises(ValueError, match=r"none can continue from 10\.0 s"):
        optimal_segments(t, y, MODELS, method="subset", candidates=[5, 10], initial={"steady"}, final={"steady"},
                         transitions={"steady": ["transient"], "transient": ["steady"]},
                         max_duration_s={"steady": 6.0}, penalty=0.1, join_transients=True)


def test_a_burst_a_billion_seconds_after_the_first_gets_the_best_joined_segmentation():
    # Ten samples, then ten more 1e9 s later: the second burst's time variance is far below the range sums'
    # rounding, so its transients are summed directly.
    t = np.concatenate((np.arange(10.0), 1e9 + np.arange(10.0)))
    rng = np.random.default_rng(2)
    y = np.concatenate((np.zeros(10), np.interp(np.arange(10.0), [0, 3, 6, 9], [0, 0, 6, 6]))) + rng.normal(0, .01, 20)
    pen = {"steady": 0.3, "transient": 0.2}
    cuts = [10, 12, 13, 15, 16, 17]
    res = optimal_segments(t, y, MODELS, method="subset", candidates=cuts, penalty=pen, join_transients=True)
    best, optima = _brute_joined(t, y, cuts=cuts, choices=PLAIN, penalty=pen)
    assert res.objective == pytest.approx(best, rel=1e-9)
    assert (_bounds(res), _names(res)) in optima
    assert any(s.state == TR and s.start >= 10 for s in res.segments)


def test_the_report_fits_a_transient_that_is_the_whole_record_with_a_free_line():
    t = np.arange(30, dtype=float)
    y = 0.2 * t + np.random.default_rng(6).normal(0, 0.05, 30)
    fit = _report(t, y, [Segment(0, 30, TR)], JOINED).blocks[0].fit
    free = fit_model(1, t, y)
    assert np.allclose(fit.fitted, free.fitted, rtol=0, atol=1e-12)
    assert "joined" not in fit.results[0].flags


def test_a_record_only_transients_could_cover_gets_the_constraints_error():
    # Transients may only follow transients, and none may be the whole record: no segmentation exists, and the
    # error is the constraints', not a missing level's.
    t, y = _step_record(0)
    with pytest.raises(ValueError, match="no segmentation satisfies the constraints") as err:
        optimal_segments(t, y, MODELS, method="exact", transitions={"transient": ["transient"]},
                         initial={"transient"}, max_duration_s={"transient": 5.0}, join_transients=True)
    assert "finite sample" not in str(err.value)


def test_a_transient_subtypes_own_line_takes_the_joined_transients_penalty():
    t, y = _step_record(0)
    subtypes = {"pulse": rs.Subtype("transient", model=1), "ramp": rs.Subtype("transient")}
    res = optimal_segments(t, y, MODELS, method="exact", subtypes=subtypes, join_transients=True)
    assert res.penalty["pulse"] == pytest.approx(res.penalty["transient"], rel=1e-12)
    assert res.penalty["steady"] == pytest.approx(2 * res.penalty["pulse"], rel=1e-12)


@pytest.mark.parametrize("y, kw, expected, alternative", [
    # A move at 6 changes the level the last transient is joined to, three boundaries on.
    ([4.7, 5.3, 5.3, 5.0, 4.3, 3.4000000000000004, 3.0, 2.7, 2.7, 2.7, 2.7, 3.3, 2.7, 3.3, 1.7, 2.3],
     dict(candidates=[1, 2, 5, 6, 8], refine_s=2.0, penalty=0.05, initial={"transient"}),
     [0, 1, 2, 6, 13, 16], [0, 1, 2, 6, 10, 16]),
    # A move two boundaries on changes what the first boundary's best position is.
    ([2.7, 3.0, 3.0, 3.3, 2.7, 2.3, 2.0, 0.7, 2.5999999999999996, 3.7, 4.7, 2.0, 2.4, 3.0999999999999996, 3.5,
      3.3000000000000003, 4.0, 0.0, 0.3],
     dict(candidates=[1, 4, 5, 8, 9, 14], refine_s=2.0, penalty=0.2, initial={"transient"}),
     [0, 5, 7, 8, 11, 16, 19], [0, 3, 7, 8, 11, 16, 19]),
])
def test_refine_searches_again_the_boundaries_a_move_can_reach(y, kw, expected, alternative):
    # A move changes a steady level, so the transients beside that segment, whose costs enter boundaries up to
    # three away. On these records the result has a lower joined objective than the alternative.
    y = np.array(y)
    t = np.arange(len(y), dtype=float)
    res = optimal_segments(t, y, MODELS, method="refine", join_transients=True, **kw)
    states = [s.state.value for s in res.segments]
    assert _bounds(res) == expected
    pen = kw["penalty"] * len(states)
    assert res.objective == pytest.approx(_joined_sse(t, y, expected, states) + pen, rel=1e-9)
    assert _joined_sse(t, y, expected, states) < _joined_sse(t, y, alternative, states)


# -- ties, and the bound that prunes ----------------------------------------------------------------------------


def _ranks(subtypes):
    """Each subtype's position among its state's subtypes."""
    names = list(subtypes)
    return {k: sum(subtypes[o].state == sub.state for o in names[:i]) for i, (k, sub) in enumerate(subtypes.items())}


OVERLAPPING = {"lo": rs.Subtype("steady", level=(None, 1.5)), "any": rs.Subtype("steady"),
               "up": rs.Subtype("transient"), "ramp": rs.Subtype("transient")}


@pytest.mark.parametrize("y", [[2, 0, 2, 1, 1, 1, 0, 2], [0, 1, 1, 1, 0, 0], [1, 1, 0, 2, 2, 2, 1]])
@pytest.mark.parametrize("order", ["declared", "reversed"])
def test_ties_go_to_the_least_rank_sum_then_the_fewest_transients(y, order):
    # Overlapping subtypes on integer data: many segmentations tie exactly on the objective.
    y = np.array(y, dtype=float)
    t = np.arange(len(y), dtype=float)
    subtypes = dict(OVERLAPPING) if order == "declared" else dict(reversed(list(OVERLAPPING.items())))
    res = optimal_segments(t, y, MODELS, method="exact", subtypes=subtypes, penalty=0.5, join_transients=True)
    choices = {k: sub.state.value for k, sub in subtypes.items()}
    accepts = {"lo": lambda v: np.isfinite(v).any() and np.nanmean(v) < 1.5}
    best, optima = _brute_joined(t, y, cuts=range(1, len(y)), choices=choices, accepts=accepts,
                                 penalty={k: 0.5 for k in subtypes})
    assert res.objective == pytest.approx(best, rel=1e-9)
    rank = _ranks(subtypes)
    key = [(sum(rank[c] for c in names), sum(choices[c] == "transient" for c in names)) for _, names in optima]
    mine = (sum(rank[c] for c in _names(res)), sum(s.state == TR for s in res.segments))
    assert len(set(key)) > 1  # the tie rule decides here
    assert mine == min(key)


def test_an_exact_tie_goes_to_the_segmentation_with_fewer_transients():
    # Flat data at zero penalties: one steady segment and a steady segment with a flat transient after it both cost
    # nothing; the joined program returns the one without the transient, as the program without the option does.
    t = np.arange(8, dtype=float)
    y = np.zeros(8)
    pen = {"steady": 0.0, "transient": 0.0}
    res = optimal_segments(t, y, MODELS, method="exact", penalty=pen, join_transients=True)
    assert [(s.start, s.stop, s.state) for s in res.segments] == [(0, 8, ST)]
    assert [(s.start, s.stop, s.state) for s in optimal_segments(t, y, MODELS, method="exact", penalty=pen).segments] \
        == [(0, 8, ST)]


def test_two_transient_options_never_follow_each_other():
    # up -> down is allowed, but both are transients: no segmentation exists, and the error is the constraints'.
    t = np.arange(12, dtype=float)
    y = np.interp(t, [0, 3, 6, 9, 11], [0, 3, 6, 3, 0])
    subtypes = {"up": rs.Subtype("transient"), "down": rs.Subtype("transient"), "lvl": rs.Subtype("steady")}
    with pytest.raises(ValueError, match="no segmentation satisfies the constraints") as err:
        optimal_segments(t, y, MODELS, method="exact", subtypes=subtypes, penalty=0.1, initial={"up"},
                         final={"lvl"}, transitions={"up": ["down"], "down": ["lvl"], "lvl": ["up"]},
                         join_transients=True)
    assert "finite sample" not in str(err.value)


def test_the_bound_that_prunes_leaves_room_for_rounding():
    # The fits apart choose the optimum here, so the bound equals its objective up to rounding; without room for
    # rounding, the search would prune the optimum itself.
    t = 1e3 + 0.1 * np.arange(10)
    y = np.round(np.array([-1.0, -0.7, -0.9, -0.9, -0.4, -0.1, 0.0, 0.1, -1.2, -1.3]), 1) + 0.1
    res = optimal_segments(t, y, MODELS, method="subset", candidates=[2, 5, 8], penalty=0.0, join_transients=True)
    best, optima = _brute_joined(t, y, cuts=[2, 5, 8], choices=PLAIN, penalty={"steady": 0.0, "transient": 0.0})
    assert res.objective == pytest.approx(best, rel=1e-9)
    assert (_bounds(res), _names(res)) in optima
    assert _names(res) == ("steady", "transient", "steady", "steady")


def test_of_equal_ways_into_a_steady_segment_the_least_rank_sum_wins():
    # Zero penalties on integer data: the steady segments before the last transient tie in many ways, and so do the
    # two transients that may lead to z, whose ranks differ. The least rank sum takes x, not w.
    t = np.arange(7, dtype=float)
    y = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    subtypes = {"p": rs.Subtype("steady"), "q": rs.Subtype("steady"), "z": rs.Subtype("steady"),
                "x": rs.Subtype("transient"), "w": rs.Subtype("transient")}
    trans = {"p": ["p", "q", "x", "w"], "q": ["p", "q", "x", "w"], "x": ["z", "p"], "w": ["z", "q"], "z": ["z", "x"]}
    kw = dict(transitions=trans, initial={"x", "p"}, final={"z"})
    res = optimal_segments(t, y, MODELS, method="exact", subtypes=subtypes, penalty=0.0,
                           max_duration_s={"p": 2.0, "q": 1.0}, join_transients=True, **kw)
    choices = {k: sub.state.value for k, sub in subtypes.items()}
    lengths = {"p": 2.0, "q": 1.0}
    best, optima = _brute_joined(t, y, cuts=range(1, 7), choices=choices, penalty={k: 0.0 for k in subtypes},
                                 accepts={k: (lambda v, d=d: len(v) <= d) for k, d in lengths.items()}, **kw)
    rank = _ranks(subtypes)
    key = [(sum(rank[c] for c in names), sum(choices[c] == "transient" for c in names)) for _, names in optima]
    assert res.objective == pytest.approx(best, abs=1e-12)
    assert len(set(key)) > 1
    assert (sum(rank[c] for c in _names(res)), sum(s.state == TR for s in res.segments)) == min(key)
    assert "w" not in _names(res)


@pytest.mark.parametrize("join", [False, True])
def test_the_objective_is_the_sse_and_every_charge_the_result_reports(join):
    # A transient starts the record: with "bic" and the option, its free slope is charged beyond its penalty, and
    # info["free_parameter_cost"] reports that, so the objective is the sum of what the result reports.
    t = np.arange(40, dtype=float)
    y = np.where(t < 10, 0.5 * t, 5.0) + np.random.default_rng(1).normal(0, 0.05, 40)
    res = optimal_segments(t, y, MODELS, method="exact", initial={"transient"}, join_transients=join)
    assert res.segments[0].state == TR
    charged = sum(res.penalty[s.subtype or s.state.value] for s in res.segments)
    extra = res.info["free_parameter_cost"]
    assert res.objective == pytest.approx(res.sse + charged + res.info["overflow_cost"] + extra, rel=1e-12)
    assert extra == (pytest.approx(res.penalty["transient"], rel=1e-12) if join else 0.0)

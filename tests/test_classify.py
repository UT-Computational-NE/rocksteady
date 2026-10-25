"""classify_segments: subtypes for segments the analyst already has, by the optimiser's definitions (#66)."""

import numpy as np
import pandas as pd
import pytest

import rocksteady as rs
from rocksteady.states import Segment, State

ST, TR = State.STEADY, State.TRANSIENT

LEVELS = {"cold": rs.Subtype("steady", level=(None, 2.5)), "hot": rs.Subtype("steady", level=(2.5, None))}
SEQUENCE = {"cold": ["heat-up"], "heat-up": ["hot"], "hot": ["cool-down"], "cool-down": ["cold"]}


def _operating_day(seed=60, n=100):
    """Cold, a rise to hot, hot, a fall back to cold, cold."""
    rng = np.random.default_rng(seed)
    t = np.arange(float(n))
    y = np.interp(t, [0, 20, 30, 60, 70, n], [0.0, 0.0, 5.0, 5.0, 0.0, 0.0]) + rng.normal(0, 0.05, n)
    return t, y


def _by_hand(y, segments):
    """A hand-written classifier: a plateau by its mean, a transient by the plateaus either side of it."""
    level = ["cold" if np.mean(y[s.slice]) < 2.5 else "hot" if s.state == ST else None for s in segments]
    out = []
    for i, s in enumerate(segments):
        if s.state == ST:
            out.append(level[i])
        else:
            out.append({("cold", "hot"): "heat-up", ("hot", "cold"): "cool-down"}[(level[i - 1], level[i + 1])])
    return out


def test_types_a_detectors_segments_as_a_hand_written_classifier_would():
    t, y = _operating_day()
    # Boundaries a detector might give: the transients wider than the ramps, the plateaus' edges not at the optimum.
    segments = [Segment(0, 17, ST), Segment(17, 36, TR), Segment(36, 58, ST, forced=True), Segment(58, 75, TR),
                Segment(75, 100, ST)]
    subtypes = {**LEVELS, "heat-up": rs.Subtype("transient"), "cool-down": rs.Subtype("transient")}
    got = rs.classify_segments(t, y, segments, subtypes=subtypes, transitions=SEQUENCE)
    assert [s.subtype for s in got] == _by_hand(y, segments) == ["cold", "heat-up", "hot", "cool-down", "cold"]
    assert [(s.start, s.stop, s.state, s.forced) for s in got] == [(s.start, s.stop, s.state, s.forced)
                                                                   for s in segments]


def _random_subtypes(rng):
    """Some of: level subtypes that split, overlap or leave a gap, peak, rise and event conditions, catch-alls."""
    a = float(rng.uniform(0.5, 2.5))
    pool = [
        {"cold": rs.Subtype("steady", level=(None, a)), "hot": rs.Subtype("steady", level=(a, None))},
        {"hot": rs.Subtype("steady", level=(a - 0.6, None)), "cold": rs.Subtype("steady", level=(None, a))},
        {"cold": rs.Subtype("steady", level=(None, a - 0.4)), "hot": rs.Subtype("steady", level=(a + 0.4, None)),
         "lukewarm": rs.Subtype("steady")},
        {"hot": rs.Subtype("steady", peak=(a, None), events={"valve": False}), "cold": rs.Subtype("steady")},
        {"up": rs.Subtype("transient", peak=(a, None)), "down": rs.Subtype("transient"),
         "other": rs.Subtype("transient")},
        {"fast": rs.Subtype("transient", max_rise_s=float(rng.integers(0, 3))), "slow": rs.Subtype("transient")},
        {"heat_up": rs.Subtype("transient", events={"heater": True}),
         "drift": rs.Subtype("transient", events={"heater": False})},
        {},  # transient without subtypes
    ]
    steady = pool[int(rng.integers(0, 4))] if rng.random() < 0.8 else {}
    return {**steady, **pool[int(rng.integers(4, 8))]}


def _random_constraints(rng, names):
    """Transitions, initial and final naming states and subtypes; caps, durations and penalties on states only."""

    def pick(k):
        return sorted(set(rng.choice(names, int(k), replace=True).tolist()))

    kw = {}
    if rng.random() < 0.7:
        kw["transitions"] = {a: pick(rng.integers(1, 4)) for a in pick(rng.integers(2, 6))}
    for edge in ("initial", "final"):
        if rng.random() < 0.5:
            kw[edge] = pick(rng.integers(1, 3))
    if rng.random() < 0.5:
        kw["per"] = 4.0
        kw["max_counts"] = {"steady": 2}
    if rng.random() < 0.4:
        kw["min_duration_s"] = {"transient": float(rng.integers(1, 3))}
    kw["penalty"] = {"steady": float(rng.choice([0.0, 0.3, 1.0])), "transient": float(rng.choice([0.0, 0.3, 1.0]))}
    return kw


@pytest.mark.parametrize("seed", range(80))
def test_on_the_optimisers_own_segments_it_returns_the_optimisers_subtypes(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(6, 10))
    t = np.arange(n, dtype=float)
    y = np.where(t < rng.integers(2, 5), 0.0, 3.0) + rng.normal(0, 0.3, n)
    if seed % 5 == 4:
        y = np.round(y)  # repeated values: exact ties between segmentations
    events = {"heater": rng.random(n) < 0.3, "valve": rng.random(n) < 0.3}
    subtypes = _random_subtypes(rng)
    kw = _random_constraints(rng, ["steady", "transient", *subtypes])
    sequence = {k: kw[k] for k in ("transitions", "initial", "final") if k in kw}
    cands = sorted(rng.choice(n, 4))
    method = [dict(method="exact"), dict(method="subset", candidates=cands),
              dict(method="refine", candidates=cands, refine_s=2.0)][seed % 3]
    try:
        r = rs.optimal_segments(t, y, {"steady": 0, "transient": 1}, subtypes=subtypes, events=events, **method, **kw)
    except ValueError as e:
        assert "no segmentation satisfies the constraints" in str(e)
        return
    got = rs.classify_segments(t, y, r.segments, subtypes=subtypes, events=events, **sequence)
    assert [s.subtype for s in got] == [s.subtype for s in r.segments]


def test_a_segmentation_that_cannot_be_typed_raises_naming_where():
    """A transient between two cold plateaus, when the only transient leads from cold to hot."""
    t = np.arange(60.0)
    y = np.interp(t, [0, 20, 30, 40, 60], [0.0, 0.0, 1.5, 0.0, 0.0])
    segments = [Segment(0, 20, ST), Segment(20, 40, TR), Segment(40, 60, ST)]
    subtypes = {**LEVELS, "heat-up": rs.Subtype("transient")}
    with pytest.raises(ValueError) as e:
        rs.classify_segments(t, y, segments, subtypes=subtypes, transitions={"cold": ["heat-up"], "heat-up": ["hot"]},
                             final=["cold", "hot"])
    assert str(e.value) == ("these segments cannot be typed: no choice of subtypes satisfies the transitions, first "
                            "and last segments and the subtypes' conditions: none can continue from 40.0 s (sample "
                            "40), in the whole record. In force: transitions={'cold': ['heat-up'], 'heat-up': "
                            "['hot']}, final=['cold', 'hot'].")
    # Without initial or final, the message does not name the first and last segments.
    with pytest.raises(ValueError) as e:
        rs.classify_segments(t, y, segments, subtypes=subtypes, transitions={"cold": ["heat-up"], "heat-up": ["hot"]})
    assert str(e.value).startswith("these segments cannot be typed: no choice of subtypes satisfies the transitions "
                                   "and the subtypes' conditions: none can continue from 40.0 s (sample 40), in the "
                                   "whole record. In force: ")
    # A catch-all transient may follow cold and lead back to it.
    subtypes["other"] = rs.Subtype("transient")
    got = rs.classify_segments(t, y, segments, subtypes=subtypes,
                               transitions={"cold": ["heat-up", "other"], "heat-up": ["hot"], "other": ["cold"]})
    assert [s.subtype for s in got] == ["cold", "other", "cold"]


def test_a_state_without_subtypes_keeps_none_and_given_subtypes_are_replaced():
    t, y = _operating_day()
    idx = pd.date_range("2026-03-02 06:00", periods=len(t), freq="1min")
    segments = [Segment(0, 20, ST, subtype="hot"), Segment(20, 30, TR, subtype="stale"),
                Segment(30, 60, ST), Segment(60, 70, State.INDETERMINATE), Segment(70, 100, ST)]
    got = rs.classify_segments(idx, y, segments, subtypes=LEVELS)
    assert [s.subtype for s in got] == ["cold", None, "hot", None, "cold"]
    assert [s.subtype for s in rs.classify_segments(t, y, segments, subtypes={})] == [None] * 5


def test_validation():
    t, y = _operating_day()
    full = [Segment(0, 50, ST), Segment(50, 100, TR)]
    with pytest.raises(ValueError, match="segments cover 50 of 100 samples; full coverage required"):
        rs.classify_segments(t, y, full[:1], subtypes=LEVELS)
    with pytest.raises(ValueError, match=r"segment 1 \[40, 100\) overlaps or precedes the previous segment"):
        rs.classify_segments(t, y, [Segment(0, 50, ST), Segment(40, 100, TR)], subtypes=LEVELS)
    with pytest.raises(ValueError, match="segment 1 has state 'warm', which is not a state"):
        rs.classify_segments(t, y, [Segment(0, 50, ST), Segment(50, 100, "warm")], subtypes=LEVELS)
    with pytest.raises(ValueError, match="t and y must be one-dimensional and of the same length"):
        rs.classify_segments(t, y[:-1], full, subtypes=LEVELS)
    for run in (rs.classify_segments, lambda *a, **kw: rs.optimal_segments(t, y, {"steady": 0}, method="exact", **kw)):
        with pytest.raises(ValueError, match=r"subtypes must be a mapping \{name: Subtype\}"):
            run(t, y, full, subtypes=[LEVELS["cold"]])
    with pytest.raises(ValueError, match="subtype 'steady' has the name of a state"):
        rs.classify_segments(t, y, full, subtypes={"steady": LEVELS["cold"]})
    with pytest.raises(ValueError, match="transitions names 'warm', which is not a state or a subtype"):
        rs.classify_segments(t, y, full, subtypes=LEVELS, transitions={"cold": ["warm"]})
    with pytest.raises(ValueError, match="initial is empty"):
        rs.classify_segments(t, y, full, subtypes=LEVELS, initial=[])
    with pytest.raises(ValueError, match="subtype 'driven' needs event mask 'heater', which events does not include"):
        rs.classify_segments(t, y, full, subtypes={"driven": rs.Subtype("transient", events={"heater": True})})
    # Every state may be named, whether or not a segment has it: there are no models here.
    sequence = {"indeterminate": ["cold"], "cold": ["transient"]}
    got = rs.classify_segments(t, y, full, subtypes=LEVELS, transitions=sequence)
    assert [s.subtype for s in got] == ["cold", None]


def test_ties_go_to_the_least_sum_of_positions_not_to_the_first_segment():
    """Steady a (0) then transient e (2) sums to 2; b (1) then c (0) sums to 1 and wins, as in the optimiser."""
    t, y = np.arange(20.0), np.zeros(20)
    segments = [Segment(0, 10, ST), Segment(10, 20, TR)]
    subtypes = {name: rs.Subtype(state) for name, state in
                [("a", "steady"), ("b", "steady"), ("c", "transient"), ("d", "transient"), ("e", "transient")]}
    sequence = {"a": ["e"], "b": ["c"]}
    got = rs.classify_segments(t, y, segments, subtypes=subtypes, transitions=sequence)
    assert [s.subtype for s in got] == ["b", "c"]
    r = rs.optimal_segments(t, y, {"steady": 0, "transient": 0}, method="subset", candidates=[0, 10, 20], penalty=0,
                            subtypes=subtypes, transitions=sequence, initial=["steady"], final=["transient"])
    assert [s.subtype for s in r.segments] == ["b", "c"]


def test_a_subtype_cap_binds_the_optimiser_but_not_classification():
    """Decided on #66: a subtype is its conditions and neighbours; caps choose the optimiser's segmentation."""
    t = np.arange(100.0)
    y = (np.interp(t, [0, 20, 25, 45, 50, 70, 75, 100], [0, 0, 5, 5, 0, 0, 5, 5])
         + np.random.default_rng(1).normal(0, 0.05, 100))
    subtypes = {**LEVELS, "other": rs.Subtype("steady")}
    r = rs.optimal_segments(t, y, {"steady": 0, "transient": 1}, method="exact", grid_s=5, penalty=1.0,
                            subtypes=subtypes, transitions={"steady": ["transient"], "transient": ["steady"]},
                            max_counts={"hot": 1})
    assert [s.subtype for s in r.segments if s.state == ST] == ["cold", "other", "cold", "hot"]
    got = rs.classify_segments(t, y, r.segments, subtypes=subtypes)
    assert [s.subtype for s in got if s.state == ST] == ["cold", "hot", "cold", "hot"]


def _walk(seed, n=60):
    rng = np.random.default_rng(seed)
    t = np.arange(float(n))
    return t, np.cumsum(rng.normal(0, 0.4, n)), sorted(rng.choice(np.arange(1, n), 5, replace=False).tolist())


WALK_SUBTYPES = {"cold": rs.Subtype("steady", level=(None, 2.5)), "hot": rs.Subtype("steady", level=(2.0, None)),
                 "up": rs.Subtype("transient", peak=(2.5, None)), "other": rs.Subtype("transient")}


def test_refine_retypes_a_moved_segment_that_an_earlier_subtype_now_accepts():
    """Decided on #66: moved to [34, 53), the middle transient reaches 2.5, so it takes up, declared first."""
    t, y, cands = _walk(16)
    r = rs.optimal_segments(t, y, {"steady": 0, "transient": 1}, method="refine", candidates=cands, refine_s=6,
                            penalty=1.0, subtypes=WALK_SUBTYPES)
    assert [(s.start, s.stop, s.subtype) for s in r.segments] == [(0, 34, "other"), (34, 53, "up"), (53, 60, "up")]


@pytest.mark.parametrize("seed", range(60))
def test_on_refines_own_segments_it_returns_refines_subtypes(seed):
    t, y, cands = _walk(seed)
    r = rs.optimal_segments(t, y, {"steady": 0, "transient": 1}, method="refine", candidates=cands, refine_s=6,
                            penalty=1.0, subtypes=WALK_SUBTYPES)
    got = rs.classify_segments(t, y, r.segments, subtypes=WALK_SUBTYPES)
    assert [s.subtype for s in got] == [s.subtype for s in r.segments]


THREE_TRANSIENTS = 11.228369548678318  # the objective of [0, 34), [34, 53) and [53, 60), each with penalty 1


@pytest.mark.parametrize("constraint, want, objective", [
    (dict(max_counts={"up": 1}), [(0, 34, "other"), (34, 53, "other"), (53, 60, "up")], THREE_TRANSIENTS),
    (dict(transitions={"steady": ["transient"], "other": ["steady", "up"], "up": ["steady"]}),
     [(0, 18, "other"), (18, 34, "cold"), (34, 53, "other"), (53, 60, "up")], 10.942807383138128),
    (dict(final=["steady", "other"]), [(0, 34, "other"), (34, 53, "up"), (53, 60, "other")], THREE_TRANSIENTS),
    (dict(max_duration_s={"up": 10.0}), [(0, 34, "other"), (34, 53, "other"), (53, 60, "up")], THREE_TRANSIENTS),
    (dict(penalty={"steady": 1.0, "transient": 1.0, "up": 3.0}),  # up would cost 2 more per segment
     [(0, 34, "other"), (34, 53, "other"), (53, 60, "other")], THREE_TRANSIENTS),
], ids=["cap", "transitions", "final", "max_duration", "penalty"])
def test_refine_retypes_its_segments_under_every_constraint(constraint, want, objective):
    """
    [34, 53) reaches 2.5, but up is capped at one, may not be followed by up, may not end the record, may last
    at most 10 s, or costs more than other: each segment takes the earliest subtype every constraint allows.
    """
    t, y, cands = _walk(16)
    r = rs.optimal_segments(t, y, {"steady": 0, "transient": 1}, method="refine", candidates=cands, refine_s=6,
                            subtypes=WALK_SUBTYPES, **{"penalty": 1.0, **constraint})
    assert [(s.start, s.stop, s.subtype) for s in r.segments] == want
    assert r.objective == pytest.approx(objective, rel=1e-12)

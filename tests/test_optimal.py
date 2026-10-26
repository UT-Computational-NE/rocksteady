"""Constrained optimal segmentation: best-fit segments under constraints (#45)."""

import dataclasses
import datetime
import math
import pickle
import re
import warnings
import zoneinfo

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
           subtypes=None, max_d=None, initial=None, final=None, events=None):
    """
    The best objective over every segmentation and every choice for each segment, by exhaustive depth-first
    search: a choice is a state without subtypes, or one subtype of a state that has some. It cuts a branch
    only when it breaks a constraint, never on cost. Returns (objective, optima): every segmentation within
    1e-9 * max(1, |best|) of the best objective, each as (bounds, choices); no optima when none satisfies the
    constraints.

    Constraints name states or subtypes. A choice's durations are its state's and its own (both bind), and
    its penalty is its own, else its state's. In ``transitions``, ``initial`` and ``final`` a state stands for
    every choice of the state.
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

    def expand(names):
        return {c for c, st, _ in choices if c in names or st in names}

    pairs = None if transitions is None else {(a, b) for k, v in transitions.items() for a in expand({k})
                                              for b in expand(set(v))}
    first = expand(set(initial)) if initial is not None else None
    last = expand(set(final)) if final is not None else None
    pen = {c: (penalty.get(c, penalty.get(st, 0.0)) if isinstance(penalty, dict) else penalty) for c, st, _ in choices}
    lo = {c: max((min_d or {}).get(k, 0.0) for k in (c, st)) for c, st, _ in choices}
    hi = {c: min((max_d or {}).get(k, math.inf) for k in (c, st)) for c, st, _ in choices}

    def cost(st, p, q):
        if (st, p, q) not in sse:
            sse[(st, p, q)] = _sse(models[st], t[p:q], y[p:q], st)
        return sse[(st, p, q)]

    def meets(sub, p, q):
        return sub is None or _meets_conditions(sub, t[p:q], y[p:q], _within(events, slice(p, q)))

    found = []  # (objective, path) of every complete segmentation

    def search(p, prev, counts, acc, path):
        if p == n:
            found.append((acc, path))
            return
        for q in sorted(c for c in cuts if c > p):
            for name, st, sub in choices:
                if not lo[name] - 1e-9 <= t[q - 1] - t[p] + dt <= hi[name] + 1e-9:
                    continue
                if pairs is not None and prev is not None and (prev, name) not in pairs:
                    continue
                if (p == 0 and first is not None and name not in first) or (q == n and last is not None
                                                                              and name not in last):
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
                search(q, name, new, acc + cost(st, p, q) + pen[name], path + [(p, q, name)])

    search(0, None, {}, 0.0, [])
    if not found:
        return math.inf, []
    best = min(acc for acc, _ in found)
    return best, [([0] + [q for _, q, _ in path], tuple(name for _, _, name in path))
                  for acc, path in found if acc <= best + 1e-9 * max(1.0, abs(best))]


def _segmentation(result):
    """A result's segmentation as ``_brute`` gives one: (bounds, choices), a choice being the subtype or state."""
    return [0] + [s.stop for s in result.segments], tuple(s.subtype or s.state.value for s in result.segments)


def _rank_sum(choices, subtypes):
    """The sum of the choices' positions among their states' subtypes (0 for a state without subtypes)."""
    names = list(subtypes)
    rank = {k: sum(subtypes[o].state == sub.state for o in names[:i]) for i, (k, sub) in enumerate(subtypes.items())}
    return sum(rank.get(c, 0) for c in choices)


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
    want_cost, optima = _brute(t, y, models, per=kw.get("per"), max_counts=kw.get("max_counts"),
                               transitions=kw.get("transitions"), min_d=kw.get("min_duration_s"), penalty=0.5)
    got = optimal_segments(t, y, models, method="exact", **kw)
    assert got.objective == pytest.approx(want_cost, rel=1e-9, abs=1e-9)
    assert _segmentation(got) in optima  # its breakpoints and states, not only its objective


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


@pytest.mark.parametrize("per", ["26h", "1560min", "1 days 02:00:00", "1 day 2 hours", "1D2h", "P1DT2H",
                                 pd.Timedelta(26, "h"), np.timedelta64(26, "h"), datetime.timedelta(hours=26)])
def test_each_form_of_a_period_names_the_same_periods(per):
    # 26-hour periods from midnight: twelve-hour segments touch three, three and two of them.
    idx = pd.date_range("2026-03-02", periods=72, freq="1h")
    segs = [Segment(i, i + 12, ST) for i in range(0, 72, 12)]
    counts = count_by_period(segs, idx, per, period_origin="2026-03-02")
    assert counts["steady"].tolist() == [3, 3, 2]
    assert list(counts.index) == [pd.Timestamp("2026-03-02 00:00"), pd.Timestamp("2026-03-03 02:00"),
                                  pd.Timestamp("2026-03-04 04:00")]


def test_a_period_string_pandas_cannot_read_is_refused_by_name():
    idx = pd.date_range("2026-03-02", periods=24, freq="1h")
    with pytest.raises(ValueError, match="'bogus'"):
        count_by_period([Segment(0, 24, ST)], idx, "bogus")


_NUMBERS = [pytest.param(7200, id="int"), pytest.param(7200.0, id="float"), pytest.param(np.int64(7200), id="np.int64"),
            pytest.param(np.float64(7200.0), id="np.float64"), pytest.param(True, id="bool"),
            pytest.param(np.True_, id="np.bool_")]
_FORMS_OF_A_PERIOD = r"a string \(e\.g\. '2h'\) or a Timedelta"


@pytest.mark.parametrize("per", _NUMBERS)
def test_a_number_as_the_period_of_datetimes_is_refused(per):
    # A number has no unit beside a DatetimeIndex; the error names the forms that have one.
    idx = pd.date_range("2026-03-02", periods=6, freq="1h")
    segs = [Segment(i, i + 1, ST) for i in range(6)]
    with pytest.raises(ValueError, match=_FORMS_OF_A_PERIOD):
        count_by_period(segs, idx, per)
    with pytest.raises(ValueError, match=_FORMS_OF_A_PERIOD):
        optimal_segments(idx, np.zeros(6), {"steady": 0}, per=per)


def _without_a_unit(numpy_time, n):
    with warnings.catch_warnings():  # numpy 2.5 deprecates making one
        warnings.simplefilter("ignore", DeprecationWarning)
        return numpy_time(n)


@pytest.mark.parametrize("per", ["7200", " 7200 ", "7200.0", "1e3",
                                 pytest.param(_without_a_unit(np.timedelta64, 7200), id="generic np.timedelta64"),
                                 pytest.param(_without_a_unit(np.timedelta64, "NaT"), id="generic np.timedelta64 NaT")])
def test_a_period_without_a_unit_is_refused(per):
    # pandas reads a bare number in a string, or a numpy timedelta without a unit, as nanoseconds.
    idx = pd.date_range("2026-03-02", periods=6, freq="1h")
    segs = [Segment(i, i + 1, ST) for i in range(6)]
    with pytest.raises(ValueError, match=r"has no unit; .*a string with one \(e\.g\. '2h'\) or a Timedelta"):
        count_by_period(segs, idx, per)
    with pytest.raises(ValueError, match="has no unit"):
        optimal_segments(idx, np.zeros(6), {"steady": 0}, per=per)


@pytest.mark.parametrize("per", [pd.NaT, "NaT", "nat", "", np.timedelta64("NaT", "h"), np.timedelta64("NaT", "ns"),
                                 pd.NA],
                         ids=["pd.NaT", "'NaT'", "'nat'", "''", "np.timedelta64 NaT h", "np.timedelta64 NaT ns",
                              "pd.NA"])
def test_a_period_pandas_reads_as_nat_is_refused_by_name(per):
    # A period that is not a duration has no period number to give a sample; the error names the value given.
    idx = pd.date_range("2026-03-02", periods=6, freq="1h")
    segs = [Segment(i, i + 1, ST) for i in range(6)]
    named = (r"per must be a duration \(e\.g\. '2h' or a Timedelta\), "
             rf"got {re.escape(repr(per))}, which pandas reads as NaT")
    with pytest.raises(ValueError, match=named):
        count_by_period(segs, idx, per)
    with pytest.raises(ValueError, match=named):
        optimal_segments(idx, np.zeros(6), {"steady": 0}, per=per)


@pytest.mark.parametrize("per", _NUMBERS)
def test_a_number_as_the_period_of_numeric_times_is_seconds(per):
    # Samples half a period apart: one-sample segments touch two per period, from 0 s (True is a 1 s period).
    step = float(per) / 2
    t = np.arange(6) * step
    segs = [Segment(i, i + 1, ST) for i in range(6)]
    counts = count_by_period(segs, t, per)
    assert counts["steady"].tolist() == [2, 2, 2]
    assert counts.index.tolist() == [0.0, 2 * step, 4 * step]


@pytest.mark.parametrize("origin", [3600, 3600.0, np.int64(3600)], ids=["int", "float", "np.int64"])
def test_a_number_as_the_period_origin_of_numeric_times_is_seconds(origin):
    # Two-hour periods from 3600 s, which is not a whole number of periods from 0 s: hourly samples fall 1, 2, 2, 1.
    t = np.arange(6) * 3600.0
    counts = count_by_period([Segment(i, i + 1, ST) for i in range(6)], t, 7200, period_origin=origin)
    assert counts["steady"].tolist() == [1, 2, 2, 1]
    assert counts.index.tolist() == [-3600.0, 3600.0, 10800.0, 18000.0]


_NUMPY_TIMES_FLOAT_READS = [
    pytest.param(np.timedelta64(3600, "ns"), id="np.timedelta64 ns"),
    pytest.param(_without_a_unit(np.timedelta64, 3600), id="generic np.timedelta64"),
    pytest.param(np.datetime64(3600, "ns"), id="np.datetime64 ns"),
]


@pytest.mark.parametrize("value", _NUMPY_TIMES_FLOAT_READS)
def test_a_numpy_time_that_is_not_nat_gives_numeric_times_the_periods_of_its_float(value):
    # float() reads each as 3600, and none is a NaT, so each keeps the periods of 3600 seconds.
    t = np.arange(6) * 3600.0
    segs = [Segment(i, i + 1, ST) for i in range(6)]
    by_origin = count_by_period(segs, t, 7200, period_origin=value)
    assert by_origin["steady"].tolist() == [1, 2, 2, 1]
    assert by_origin.index.tolist() == [-3600.0, 3600.0, 10800.0, 18000.0]
    by_per = count_by_period(segs, t, value)
    assert by_per["steady"].tolist() == [1, 1, 1, 1, 1, 1]
    assert by_per.index.tolist() == [0.0, 3600.0, 7200.0, 10800.0, 14400.0, 18000.0]
    fit = optimal_segments(t, np.zeros(6), {"steady": 0}, per=value, period_origin=value, method="exact")
    assert fit.counts.index.tolist() == [0.0, 3600.0, 7200.0, 10800.0, 14400.0, 18000.0]


_NUMPY_NATS = [np.datetime64("NaT", "ns"), np.timedelta64("NaT", "s"), _without_a_unit(np.timedelta64, "NaT")]
_NUMPY_NAT_IDS = ["np.datetime64 NaT ns", "np.timedelta64 NaT s", "generic np.timedelta64 NaT"]


@pytest.mark.parametrize("origin", [np.nan, float("nan"), np.inf, -np.inf, pd.NaT, pd.NA] + _NUMPY_NATS,
                         ids=["np.nan", "float nan", "inf", "-inf", "pd.NaT", "pd.NA"] + _NUMPY_NAT_IDS)
def test_a_period_origin_of_numeric_times_that_is_not_a_finite_number_is_refused_by_name(origin):
    # An origin that is not a finite number of seconds would put every sample in one period; the error names it.
    t = np.arange(6) * 3600.0
    segs = [Segment(i, i + 1, ST) for i in range(6)]
    named = rf"period_origin must be a finite number of seconds, got {re.escape(repr(origin))}$"
    with pytest.raises(ValueError, match=named):
        count_by_period(segs, t, 7200, period_origin=origin)
    with pytest.raises(ValueError, match=named):
        optimal_segments(t, np.zeros(6), {"steady": 0}, per=7200, period_origin=origin)


@pytest.mark.parametrize("per", [pd.NaT, pd.NA] + _NUMPY_NATS, ids=["pd.NaT", "pd.NA"] + _NUMPY_NAT_IDS)
def test_a_missing_period_of_numeric_times_is_refused_by_name(per):
    t = np.arange(6) * 3600.0
    segs = [Segment(i, i + 1, ST) for i in range(6)]
    named = rf"per must be a positive number of seconds, got {re.escape(repr(per))}$"
    with pytest.raises(ValueError, match=named):
        count_by_period(segs, t, per)
    with pytest.raises(ValueError, match=named):
        optimal_segments(t, np.zeros(6), {"steady": 0}, per=per)


_CHICAGO = zoneinfo.ZoneInfo("America/Chicago")


@pytest.mark.parametrize("origin", [
    pytest.param("2026-03-02 01:00", id="naive str"),
    pytest.param("01:00", id="naive str without a date"),  # dated today: 4 h periods from 01:00 on any day
    pytest.param(pd.Timestamp("2026-03-02 01:00"), id="naive Timestamp"),
    pytest.param(datetime.datetime(2026, 3, 2, 1), id="naive datetime"),
    pytest.param(np.datetime64("2026-03-02T01:00", "ns"), id="np.datetime64 ns"),
    pytest.param(np.datetime64("2026-03-02T01:00", "m"), id="np.datetime64 m"),
    pytest.param(pd.Timestamp("2026-03-02 01:00", tz="America/Chicago"), id="aware Timestamp"),
    pytest.param(datetime.datetime(2026, 3, 2, 1, tzinfo=_CHICAGO), id="aware datetime"),
    # The same instant in other zones: 07:00 UTC, 02:00 New York. Read on their own clocks, they would start
    # periods at 23:00 and give [3, 3], or at 02:00 and give [2, 4].
    pytest.param(pd.Timestamp("2026-03-02 07:00", tz="UTC"), id="aware Timestamp, UTC"),
    pytest.param(datetime.datetime(2026, 3, 2, 7, tzinfo=datetime.timezone.utc), id="aware datetime, UTC"),
    pytest.param(pd.Timestamp("2026-03-02 02:00", tz="America/New_York"), id="aware Timestamp, New York"),
    pytest.param("2026-03-02 07:00+00:00", id="aware str, UTC"),
    pytest.param("2026-03-02T07:00Z", id="aware str, Z"),
    pytest.param("Mon, 02 Mar 2026 07:00:00 +0000", id="aware str, RFC 2822"),
    pytest.param("2026-03-02 01:00-06:00", id="aware str, Chicago"),
])
def test_a_period_origin_is_read_on_the_wall_clock(origin):
    # Four-hour periods from 01:00 local, whether the origin is naive or an aware time in any zone. Chicago is
    # UTC-6, which is not a whole number of periods: read in UTC, the origin would start them at 23:00 and give
    # [3, 3].
    idx = pd.date_range("2026-03-02", periods=6, freq="1h", tz="America/Chicago")
    segs = [Segment(i, i + 1, ST) for i in range(6)]
    starts = [pd.Timestamp("2026-03-01 21:00"), pd.Timestamp("2026-03-02 01:00"), pd.Timestamp("2026-03-02 05:00")]
    counts = count_by_period(segs, idx, "4h", period_origin=origin)
    assert counts["steady"].tolist() == [1, 4, 1]
    assert list(counts.index) == starts
    # The cap binds per period: one segment in each of the three periods.
    result = optimal_segments(idx, np.arange(6.0), {"steady": 0}, per="4h", period_origin=origin,
                              max_counts={"steady": 1}, method="exact")
    assert [(s.start, s.stop) for s in result.segments] == [(0, 1), (1, 5), (5, 6)]
    assert list(result.counts.index) == starts


@pytest.mark.parametrize("origin", ["2026-07-01 07:00+00:00", pd.Timestamp("2026-07-01 07:00", tz="UTC")],
                         ids=["str", "Timestamp"])
def test_an_aware_period_origin_is_converted_at_its_own_instant_whatever_the_records_season(origin):
    # 07:00 UTC on 1 July is 02:00 in Chicago (daylight saving time, UTC-5), so four-hour periods start at 22:00 and
    # 02:00 beside a March record, and hourly samples from midnight fall [2, 4]. Converted on the record's date
    # (01:00, UTC-6) they would give [1, 4, 1], and read on its own clock (07:00) [3, 3].
    idx = pd.date_range("2026-03-02", periods=6, freq="1h", tz="America/Chicago")
    segs = [Segment(i, i + 1, ST) for i in range(6)]
    starts = [pd.Timestamp("2026-03-01 22:00"), pd.Timestamp("2026-03-02 02:00")]
    counts = count_by_period(segs, idx, "4h", period_origin=origin)
    assert counts["steady"].tolist() == [2, 4]
    assert list(counts.index) == starts
    result = optimal_segments(idx, np.arange(6.0), {"steady": 0}, per="4h", period_origin=origin,
                              max_counts={"steady": 1}, method="exact")
    assert [(s.start, s.stop) for s in result.segments] == [(0, 2), (2, 6)]
    assert list(result.counts.index) == starts


def test_an_aware_period_origin_beside_a_naive_index_keeps_its_own_clock():
    # A naive index has no zone to convert to: 02:00 in New York starts periods at 02:00 on the index's clock, so
    # four-hour periods start at 22:00 and 02:00 and hourly samples from midnight fall [2, 4]. Read in UTC (07:00),
    # they would start at 23:00 and give [3, 3].
    idx = pd.date_range("2026-03-02", periods=6, freq="1h")
    counts = count_by_period([Segment(i, i + 1, ST) for i in range(6)], idx, "4h",
                             period_origin=pd.Timestamp("2026-03-02 02:00", tz="America/New_York"))
    assert counts["steady"].tolist() == [2, 4]
    assert list(counts.index) == [pd.Timestamp("2026-03-01 22:00"), pd.Timestamp("2026-03-02 02:00")]


@pytest.mark.parametrize("origin", _NUMBERS)
def test_a_number_as_the_period_origin_of_datetimes_is_refused(origin):
    # A number has no unit or epoch beside a DatetimeIndex; the error names the timestamp form.
    idx = pd.date_range("2026-03-02", periods=6, freq="1h")
    segs = [Segment(i, i + 1, ST) for i in range(6)]
    with pytest.raises(ValueError, match=r"period_origin as a number .* a timestamp \(e\.g\. '2026-03-02 06:00'\)"):
        count_by_period(segs, idx, "2h", period_origin=origin)
    with pytest.raises(ValueError, match=r"period_origin as a number .* a timestamp"):
        optimal_segments(idx, np.zeros(6), {"steady": 0}, per="2h", period_origin=origin)


@pytest.mark.parametrize("origin", [pd.NaT, "NaT", "nat", "", np.datetime64("NaT", "ns"),
                                    _without_a_unit(np.datetime64, "NaT"), pd.NA],
                         ids=["pd.NaT", "'NaT'", "'nat'", "''", "np.datetime64 NaT ns", "generic np.datetime64 NaT",
                              "pd.NA"])
def test_a_period_origin_pandas_reads_as_nat_is_refused_by_name(origin):
    # An origin that is not a time puts no sample in a period; the error names the value given.
    idx = pd.date_range("2026-03-02", periods=6, freq="1h")
    segs = [Segment(i, i + 1, ST) for i in range(6)]
    named = (r"period_origin must be a timestamp \(e\.g\. '2026-03-02 06:00'\), "
             rf"got {re.escape(repr(origin))}, which pandas reads as NaT")
    with pytest.raises(ValueError, match=named):
        count_by_period(segs, idx, "2h", period_origin=origin)
    with pytest.raises(ValueError, match=named):
        optimal_segments(idx, np.zeros(6), {"steady": 0}, per="2h", period_origin=origin)


@pytest.mark.parametrize("tz", ["America/Chicago", None], ids=["aware index", "naive index"])
@pytest.mark.parametrize("origin", ["07:00+00:00", "1:00-06:00", "7am UTC", "T07:00Z", "Mar 2 07:00+00:00",
                                    "2026-03 07:00+00:00"])
def test_an_aware_period_origin_without_a_whole_date_is_refused_by_name(origin, tz):
    # pandas dates a time without a date today, so its periods would shift with the season it runs in, or in year 1
    # (pandas 1.5 cannot), and a month without a day on the 1st.
    idx = pd.date_range("2026-03-02", periods=6, freq="1h", tz=tz)
    segs = [Segment(i, i + 1, ST) for i in range(6)]
    named = (r"an aware period_origin needs a whole date \(e\.g\. '2026-03-02 07:00\+00:00'\), "
             rf"got {re.escape(repr(origin))}$")
    with pytest.raises(ValueError, match=named):
        count_by_period(segs, idx, "4h", period_origin=origin)
    with pytest.raises(ValueError, match=named):
        optimal_segments(idx, np.zeros(6), {"steady": 0}, per="4h", period_origin=origin)


def test_count_tables_without_segments_or_columns_are_empty():
    idx = pd.date_range("2026-03-02", periods=48, freq="1h")
    no_segments = count_by_period([], idx, "1D", states=["steady"])
    assert no_segments.shape == (0, 1) and list(no_segments.columns) == ["steady"]
    assert (no_segments.dtypes == "int64").all() and no_segments.index.name == "period"
    assert count_by_period([Segment(0, 48, ST)], idx, "1D", states=[]).shape == (0, 0)
    assert count_by_period([Segment(0, 48, ST)], np.arange(48.0), 25.0, states=[]).shape == (0, 0)


@pytest.mark.parametrize("datetimes", [False, True])
def test_a_table_of_some_states_holds_only_the_periods_they_touch(datetimes):
    t = pd.date_range("2026-03-02", periods=30, freq="1h") if datetimes else np.arange(30.0)
    per = "10h" if datetimes else 10.0
    segs = [Segment(0, 10, TR), Segment(10, 20, ST), Segment(20, 30, TR)]
    kw = dict(period_origin=t[0]) if datetimes else {}
    steady = count_by_period(segs, t, per, states=["steady"], **kw)
    assert steady["steady"].tolist() == [1] and steady.index[0] == (t[10] if datetimes else 10.0)
    assert count_by_period(segs, t, per, states=["indeterminate"], **kw).shape == (0, 1)


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
    t, y, _ = _startup_record(n=900)
    models = {"steady": 0, "transient": 1}
    kw = dict(method="exact", grid_s=10, per=300.0, max_counts={"steady": 2, "transient": 2}, penalty=5.0)
    a = optimal_segments(t, y, models, **kw)
    b = optimal_segments(t + 3e8, y, models, period_origin=3e8, **kw)
    assert [(s.start, s.stop, s.state) for s in a.segments] == [(s.start, s.stop, s.state) for s in b.segments]
    assert b.objective == pytest.approx(a.objective, rel=1e-6)


# -- pruning, cost precision and clock changes -----------------------------------------


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


@pytest.mark.parametrize("kw, opening", [
    ({}, "(counts, durations, transitions) on"),
    ({"subtypes": {}}, "(counts, durations, transitions) on"),
    ({"subtypes": {"cold": rs.Subtype("steady")}}, "(counts, durations, transitions) and the subtypes' conditions on"),
    ({"initial": ["steady"]}, "(counts, durations, transitions, first and last segments) on"),
    ({"final": ["steady"]}, "(counts, durations, transitions, first and last segments) on"),
    ({"subtypes": {"cold": rs.Subtype("steady")}, "final": ["cold"]},
     "(counts, durations, transitions, first and last segments) and the subtypes' conditions on"),
    # Subtypes of one state, beside a state without subtypes (here one that can never be placed).
    ({"subtypes": {"cold": rs.Subtype("steady")}, "models": {"steady": 0, "transient": 1},
      "max_counts": {"steady": 1, "transient": 0}},
     "(counts, durations, transitions) and the subtypes' conditions on"),
    # Subtypes of a later state only.
    ({"subtypes": {"tr": rs.Subtype("transient")}, "models": {"steady": 0, "transient": 1},
      "max_counts": {"steady": 1, "transient": 0}},
     "(counts, durations, transitions) and the subtypes' conditions on"),
])
def test_an_infeasible_call_names_only_what_it_uses(kw, opening):
    # One segment of at most 4 s cannot cover 10 s. The message names the first and last segments only when
    # initial or final is given, and the subtypes' conditions only when there are subtypes.
    t = np.arange(10.0)
    kw = {"models": {"steady": 0}, "max_counts": {"steady": 1}, **kw}
    msg = _infeasible_message(t, np.zeros(10), method="exact", max_duration_s={"steady": 4.0}, penalty=0.0, **kw)
    assert msg.startswith(f"no segmentation satisfies the constraints {opening} these candidate breakpoints: none "
                          "can continue from 4.0 s (sample 4), in the whole record. In force: ")


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
    # A transient capped at 0 cannot be placed, so its exp model is never fitted.
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
                              per=pd.Timedelta(1, "D"), max_counts={"steady": np.int64(1), "transient": np.int64(0)},
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
    want_cost, optima = _brute(t, y, models, penalty=0.5, subtypes=subtypes, **kw)
    got = optimal_segments(t, y, models, method="exact", penalty=0.5, subtypes=subtypes, **kw)
    assert got.objective == pytest.approx(want_cost, rel=1e-9, abs=1e-9)
    # Its breakpoints and each segment's subtype: an optimum, and of equal objectives the least rank sum.
    assert _segmentation(got) in optima
    assert _rank_sum(_segmentation(got)[1], subtypes) == min(_rank_sum(names, subtypes) for _, names in optima)
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
                                   ([0.1, 0.3, 0.2], 0.2),
                                   # samples of other magnitudes than the threshold, below and above it
                                   ([1.0, 3.0], 2.0), ([-4.0, 8.0], 2.0),
                                   # means that need every bit of the samples' and the threshold's mantissas
                                   ([1 + 2 ** -52, 1 + 3 * 2 ** -52], 1 + 2 * 2 ** -52),
                                   ([2 ** -1074, 3 * 2 ** -1074], 2 ** -1073),
                                   # an odd count times a full mantissa, which a float would round
                                   ([1.0, 1 + 2 ** -52, 1 + 2 * 2 ** -52], 1 + 2 ** -52)])
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


def test_a_mean_just_below_a_threshold_on_a_long_segment_takes_the_lower_subtype():
    """
    1999 samples of 0.7 and one a step below it: the mean is below 0.7 by about 6e-20. The count times
    0.7's 53-bit mantissa is beyond a 64-bit integer, so the comparison must not be made in one.
    """
    y = np.array([0.7] * 1999 + [np.nextafter(0.7, 0.0)])
    t = np.arange(float(len(y)))
    subtypes = {"cold": rs.Subtype("steady", level=(None, 0.7)), "hot": rs.Subtype("steady", level=(0.7, None))}
    r = optimal_segments(t, y, {"steady": 0}, method="subset", candidates=[0, len(y)], penalty=0, subtypes=subtypes)
    assert not _exact_level_ok(y, (0.7, None))
    assert [s.subtype for s in r.segments] == ["cold"]


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


@pytest.mark.parametrize("named", [False, True])
def test_pruning_stays_exact_with_subtypes(named):
    """
    Several options per state, conditions that reject segments, and a cap on a state and on its subtype; and
    (``named``) subtypes' own penalties and durations, transitions between subtypes, and the first and last.
    """
    t, y, _ = _startup_record(n=900)
    models = {"steady": 0, "transient": {"model": "first_order", "tau_min": 1.0, "tau_max": 1e3}}
    subtypes = {"zero": rs.Subtype("steady", level=(None, 5.0)), "power": rs.Subtype("steady", level=(60.0, None)),
                "rising": rs.Subtype("transient", level=(None, 50.0)), "other": rs.Subtype("transient")}
    extra = dict(penalty={"transient": 50.0, "other": 5.0, "zero": 20.0}, max_duration_s={"rising": 300.0},
                 min_duration_s={"transient": 30.0, "other": 60.0}, initial={"zero"}, final={"steady", "other"},
                 transitions={"zero": ["rising"], "rising": ["power"], "power": ["other"], "other": ["steady"]}
                 ) if named else {}
    for kw in (dict(method="exact", grid_s=30), dict(method="subset", candidates=list(range(0, 901, 25)))):
        a, b = (optimal_segments(t, y, models, subtypes=subtypes, per=450.0, prune=prune,
                                 max_counts={"steady": 2, "power": 1, "transient": 2}, **kw, **extra)
                for prune in (True, False))
        assert a.objective == pytest.approx(b.objective, rel=1e-12)
        assert [(s.start, s.stop, s.subtype) for s in a.segments] == [(s.start, s.stop, s.subtype) for s in b.segments]
        if kw["method"] == "exact":
            assert {s.subtype for s in a.segments} >= {"zero", "power", "rising", "other"}


def test_no_subtypes_gives_the_same_result():
    t, y, _ = _startup_record(n=900)
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
    for pair in (np.array([1.0, 2.0]), pd.Series([1.0, 2.0]), [1.0, 2.0]):  # any ordered pair
        assert rs.Subtype("steady", level=pair).level == (1.0, 2.0)
    for level in [(5.0, 1.0), (1.0, 1.0)]:
        with pytest.raises(ValueError, match=r"level lo \(\S+\) must be < Subtype level hi"):
            rs.Subtype("steady", level=level)
    # a number, ragged pairs (which numpy cannot make an array of), unordered pairs, a 2 x 2 array
    # a pair with no length (an iterator, a generator, a 0-d array), too few or too many items, bytes
    for level in (40.0, (1, (2, 3)), ((1, 2), (3, 4, 5)), {1.0, 2.0}, {1.0: 2.0, 3.0: 4.0}, np.eye(2),
                  iter([1.0, 2.0]), (v for v in (1.0, 2.0)), np.array(40.0), (), (1.0,), (1.0, 2.0, 3.0), b"ab", "ab"):
        with pytest.raises(ValueError, match=re.escape(f"Subtype level must be a pair (lo, hi), got {level!r}")):
            rs.Subtype("steady", level=level)
    with pytest.raises(ValueError, match="Subtype peak must be a pair"):
        rs.Subtype("steady", peak=(1, (2, 3)))
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
    with pytest.raises(ValueError, match="penalty names 'warm', which is not a state with a model or a subtype"):
        run({"hot": rs.Subtype("steady")}, penalty={"warm": 5.0})
    with pytest.raises(ValueError, match="subtypes' conditions"):
        run({"hot": rs.Subtype("steady", level=(100.0, None))})
    with pytest.raises(ValueError, match=r"max_counts=\{'hot': 0\} per period \(5.0\)"):  # a subtype's cap
        run({"hot": rs.Subtype("steady")}, max_counts={"hot": 0}, per=5.0)


def test_a_tie_between_predecessors_of_different_states_still_goes_to_the_subtype_declared_first():
    """[3, 5) takes 'low', declared first, over 'mid' at the same objective, though its predecessor's state differs."""
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
        # The cap is the rule, not a copy of the code: in exact arithmetic the threshold never exceeds the
        # maximum, and the cap only absorbs rounding. Without it, with rise_fraction=1 about 1.7 % of
        # sign-changing segments get a threshold just above their own maximum (2.3500000000000005 for 2.35),
        # which no sample reaches.
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


# -- every constraint can name subtypes; initial and final (#63) -------------------------


def _operating_day(seed=60, n=100):
    """Cold, a rise to hot, hot, a fall back to cold, cold."""
    rng = np.random.default_rng(seed)
    t = np.arange(float(n))
    y = np.interp(t, [0, 20, 30, 60, 70, n], [0.0, 0.0, 5.0, 5.0, 0.0, 0.0]) + rng.normal(0, 0.05, n)
    return t, y


_LEVELS = {"cold": rs.Subtype("steady", level=(None, 2.5)), "hot": rs.Subtype("steady", level=(2.5, None))}
_SEQUENCE = {"cold": ["heat-up"], "heat-up": ["hot"], "hot": ["cool-down"], "cool-down": ["cold"]}


@pytest.mark.parametrize("order", [("heat-up", "cool-down"), ("cool-down", "heat-up")])
def test_transient_subtypes_with_the_same_conditions_are_told_apart_by_transitions(order):
    """Heat-up and cool-down have no conditions: only their neighbours in `transitions` tell them apart."""
    t, y = _operating_day()
    subtypes = {**_LEVELS, **{name: rs.Subtype("transient") for name in order}}
    kw = dict(method="exact", grid_s=5, penalty=1.0, subtypes=subtypes)
    r = optimal_segments(t, y, {"steady": 0, "transient": 1}, transitions=_SEQUENCE, **kw)
    assert [s.subtype for s in r.segments] == ["cold", "heat-up", "hot", "cool-down", "cold"]
    alternate = {"steady": ["transient"], "transient": ["steady"]}
    free = optimal_segments(t, y, {"steady": 0, "transient": 1}, transitions=alternate, **kw)
    assert [s.subtype for s in free.segments] == ["cold", order[0], "hot", order[0], "cold"]  # a tie: the first


_BY_STATE = [  # (with state names, the same with every subtype listed)
    ({"steady": ["transient"], "transient": ["steady"]},
     {"cold": ["heat-up", "cool-down"], "hot": ["heat-up", "cool-down"], "heat-up": ["cold", "hot"],
      "cool-down": ["cold", "hot"]}),
    ({"steady": ["transient"], "transient": ["hot"], "hot": ["cool-down"], "cool-down": ["cold"]},  # a union
     {"cold": ["heat-up", "cool-down"], "hot": ["heat-up", "cool-down"], "heat-up": ["hot"],
      "cool-down": ["hot", "cold"]}),
    ({"cold": ["transient"], "transient": ["steady", "transient"], "hot": ["cool-down"]},
     {"cold": ["heat-up", "cool-down"], "heat-up": ["cold", "hot", "heat-up", "cool-down"],
      "cool-down": ["cold", "hot", "heat-up", "cool-down"], "hot": ["cool-down"]}),
]


@pytest.mark.parametrize("seed", range(2))
@pytest.mark.parametrize("case", range(len(_BY_STATE)))
def test_a_states_name_in_transitions_is_all_of_its_subtypes(seed, case):
    t, y = _operating_day(seed=seed)
    subtypes = {**_LEVELS, "heat-up": rs.Subtype("transient"), "cool-down": rs.Subtype("transient")}
    by_state, listed = _BY_STATE[case]
    kw = dict(method="exact", grid_s=5, penalty=[0.2, 1.0][seed], subtypes=subtypes, max_counts={"heat-up": 1})
    a = optimal_segments(t, y, {"steady": 0, "transient": 1}, transitions=by_state, **kw)
    b = optimal_segments(t, y, {"steady": 0, "transient": 1}, transitions=listed, **kw)
    assert [(s.start, s.subtype) for s in a.segments] == [(s.start, s.subtype) for s in b.segments]
    assert a.objective == b.objective


def _cut_off_day():
    """A day that starts partway up a rise (its cold side is missing), then hot, a fall, and cold."""
    t, y = _operating_day()
    return t[25:] - 25.0, y[25:]


@pytest.mark.parametrize("initial, first", [(None, "heat-up"), ({"steady", "other"}, "other"), (["other"], "other"),
                                            ({"cold", "transient"}, "heat-up")])
def test_a_transient_at_the_start_takes_no_subtype_initial_excludes(initial, first):
    t, y = _cut_off_day()
    subtypes = {**_LEVELS, "heat-up": rs.Subtype("transient"), "other": rs.Subtype("transient")}
    r = optimal_segments(t, y, {"steady": 0, "transient": 1}, method="exact", grid_s=5, penalty=1.0,
                         transitions={"steady": ["transient"], "transient": ["steady"]}, subtypes=subtypes,
                         initial=initial)
    assert [s.subtype for s in r.segments] == [first, "hot", "heat-up", "cold"]


@pytest.mark.parametrize("final, last", [(None, "cool-down"), ({"steady", "other"}, "other")])
def test_a_transient_at_the_end_takes_no_subtype_final_excludes(final, last):
    """The day reversed: it ends partway down a fall, whose cold side is missing."""
    t, y = _cut_off_day()
    y = y[::-1].copy()
    subtypes = {**_LEVELS, "cool-down": rs.Subtype("transient"), "other": rs.Subtype("transient")}
    r = optimal_segments(t, y, {"steady": 0, "transient": 1}, method="exact", grid_s=5, penalty=1.0,
                         transitions={"steady": ["transient"], "transient": ["steady"]}, subtypes=subtypes,
                         final=final)
    assert [s.subtype for s in r.segments] == ["cold", "cool-down", "hot", last]


@pytest.mark.parametrize("initial, final, want", [({"a", "b"}, {"b"}, "b"), ({"b"}, {"a", "b"}, "b"),
                                                  ({"a"}, {"b"}, None), ({"steady"}, {"b"}, "b")])
def test_a_one_segment_record_satisfies_both_initial_and_final(initial, final, want):
    t = np.arange(10.0)
    y = np.random.default_rng(61).normal(0, 0.1, 10)
    subtypes = {"a": rs.Subtype("steady"), "b": rs.Subtype("steady")}
    kw = dict(method="subset", candidates=[0, 10], penalty=0, subtypes=subtypes, initial=initial, final=final)
    if want is None:
        with pytest.raises(ValueError, match=r"initial=\['a'\], final=\['b'\]"):
            optimal_segments(t, y, {"steady": 0}, **kw)
    else:
        assert [s.subtype for s in optimal_segments(t, y, {"steady": 0}, **kw).segments] == [want]


def _constraint_scheme(rng):
    """Random constraints naming states and subtypes: transitions, initial and final, durations, penalties."""
    names = ["steady", "transient", "cold", "hot", "up", "down", "other"]

    def pick(k):
        return sorted(set(rng.choice(names, int(k), replace=True).tolist()))

    kw = {}
    if rng.random() < 0.7:
        kw["transitions"] = {a: pick(rng.integers(1, 4)) for a in pick(rng.integers(2, 6))}
    for edge in ("initial", "final"):
        if rng.random() < 0.6:
            kw[edge] = pick(rng.integers(1, 3))
    if rng.random() < 0.6:
        kw["min_d"] = {k: float(rng.integers(1, 4)) for k in pick(rng.integers(1, 3))}
    if rng.random() < 0.6:
        kw["max_d"] = {k: float(rng.integers(2, 6)) for k in pick(rng.integers(1, 3))}
    kw["penalty"] = {k: float(rng.choice([0.0, 0.3, 1.0, 3.0])) for k in pick(rng.integers(1, 5))}
    return kw


@pytest.mark.parametrize("seed", range(40))
def test_exact_matches_brute_force_with_constraints_naming_subtypes(seed):
    rng = np.random.default_rng(900 + seed)
    n = 7
    t = np.arange(n, dtype=float)
    y = np.where(t < rng.integers(2, 5), 0.0, 3.0) + rng.normal(0, 0.3, n)
    a = float(rng.uniform(0.5, 2.5))
    subtypes = {"cold": rs.Subtype("steady", level=(None, a)), "hot": rs.Subtype("steady", level=(a, None)),
                "up": rs.Subtype("transient", peak=(a, None)), "down": rs.Subtype("transient"),
                "other": rs.Subtype("transient")}
    models = {"steady": 0, "transient": 1}
    kw = _constraint_scheme(rng)
    caps = {"steady": 2, "up": 1} if seed % 2 else None
    per = 3.0 if seed % 3 == 0 else None
    want_cost, optima = _brute(t, y, models, per=per, max_counts=caps, subtypes=subtypes, **kw)
    args = dict(method="exact", subtypes=subtypes, per=per, max_counts=caps, penalty=kw["penalty"],
                transitions=kw.get("transitions"), initial=kw.get("initial"), final=kw.get("final"),
                min_duration_s=kw.get("min_d"), max_duration_s=kw.get("max_d"))
    if not optima:
        with pytest.raises(ValueError, match="no segmentation satisfies the constraints"):
            optimal_segments(t, y, models, **args)
        return
    got = optimal_segments(t, y, models, **args)
    assert got.objective == pytest.approx(want_cost, rel=1e-9, abs=1e-9)
    pen = {s: kw["penalty"].get(s.subtype, kw["penalty"].get(s.state.value, 0.0)) for s in got.segments}
    assert sum(_sse(models[s.state.value], t[s.slice], y[s.slice], s.state.value) + pen[s]
               for s in got.segments) == pytest.approx(got.objective, rel=1e-9, abs=1e-9)


def test_constraints_naming_subtypes_validation():
    t, y = np.arange(10.0), np.zeros(10)

    def run(**kw):
        return optimal_segments(t, y, {"steady": 0}, method="exact", subtypes={"hot": rs.Subtype("steady")}, **kw)

    for what in ("transitions", "min_duration_s", "max_duration_s", "penalty", "initial", "final"):
        bad = {"transitions": {"hot": ["warm"]}, "initial": ["warm"], "final": {"warm"}}.get(what, {"warm": 1.0})
        with pytest.raises(ValueError, match=f"{what} names 'warm', which is not a state with a model or a subtype"):
            run(**{what: bad})
        bad = {"transitions": {"transient": ["hot"]}, "initial": ["transient"], "final": {"transient"}}.get(
            what, {"transient": 1.0})
        with pytest.raises(ValueError, match=f"{what} names state 'transient', which has no model"):
            run(**{what: bad})
    for what in ("initial", "final"):
        with pytest.raises(ValueError, match=f"{what} must be a collection of state or subtype names, got 'hot'"):
            run(**{what: "hot"})
        with pytest.raises(ValueError, match=f"{what} is empty: name at least one state or subtype"):
            run(**{what: set()})
        assert [s.subtype for s in run(**{what: ("hot",)}).segments] == ["hot"]
    for what in ("min_duration_s", "max_duration_s"):
        with pytest.raises(ValueError, match=rf"{what}\['hot'\] must be >= 0"):
            run(**{what: {"hot": -1.0}})
    with pytest.raises(ValueError, match=r"penalties must be finite and >= 0, got .*'hot': -1.0"):
        run(penalty={"hot": -1.0})
    assert run(penalty={"steady": 2.0, "hot": 5.0}).penalty == {"steady": 2.0, "hot": 5.0}
    assert run(penalty=2.0).penalty == {"steady": 2.0, "hot": 2.0}
    noisy = np.random.default_rng(62).normal(0, 1.0, 10)
    pen = optimal_segments(t, noisy, {"steady": 0}, method="exact", subtypes={"hot": rs.Subtype("steady")}).penalty
    plain = optimal_segments(t, noisy, {"steady": 0}, method="exact").penalty
    assert pen == {"steady": plain["steady"], "hot": plain["steady"]} and plain["steady"] > 0  # "bic": the state's


@pytest.mark.parametrize("subtypes", [None, {**_LEVELS, "other": rs.Subtype("transient")}])
def test_initial_and_final_naming_every_state_change_nothing(subtypes):
    t, y = _operating_day()
    kw = dict(method="exact", grid_s=5, penalty=1.0, max_counts={"steady": 3, "transient": 2}, subtypes=subtypes,
              transitions={"steady": ["transient"], "transient": ["steady"]})
    every = {"steady", "transient"}
    a = optimal_segments(t, y, {"steady": 0, "transient": 1}, **kw)
    b = optimal_segments(t, y, {"steady": 0, "transient": 1}, initial=every, final=every, **kw)
    assert a.segments == b.segments and a.objective == b.objective and a.counts.equals(b.counts)


@pytest.mark.parametrize("lo, hi", [("hot", "steady"), ("steady", "hot"), ("hot", "hot")])
def test_a_subtype_whose_durations_do_not_overlap_is_refused(lo, hi):
    """A subtype whose durations and its state's leave none could never be used: it is refused up front."""
    t, y = _operating_day()
    culprit = f"min_duration_s 50.0 s ({lo}) > max_duration_s 40.0 s ({hi})"
    with pytest.raises(ValueError, match=re.escape(f"subtype 'hot' can take no segment: {culprit}")):
        optimal_segments(t, y, {"steady": 0, "transient": 1}, method="exact", grid_s=5, subtypes=dict(_LEVELS),
                         min_duration_s={lo: 50.0}, max_duration_s={hi: 40.0})


@pytest.mark.parametrize("subtypes", [None, dict(_LEVELS)])
@pytest.mark.parametrize("state", ["steady", "transient"])
def test_a_state_whose_own_durations_do_not_overlap_is_refused(state, subtypes):
    """A state's own min_duration_s above its max_duration_s is refused, whether or not it has subtypes."""
    t, y = _operating_day()
    culprit = f"min_duration_s 50.0 s ({state}) > max_duration_s 40.0 s ({state})"
    with pytest.raises(ValueError, match=re.escape(f"state {state!r} can take no segment: {culprit}")):
        optimal_segments(t, y, {"steady": 0, "transient": 1}, method="exact", grid_s=5, penalty=1.0,
                         subtypes=subtypes, min_duration_s={state: 50.0}, max_duration_s={state: 40.0})
    # Equal bounds leave one duration: accepted.
    optimal_segments(t, y, {"steady": 0, "transient": 1}, method="exact", grid_s=5, penalty=1.0,
                     subtypes=subtypes, min_duration_s={state: 40.0}, max_duration_s={state: 40.0})


@pytest.mark.parametrize("b, want", [(0.5, "b"), (1.0, "a"), (1.5, "a")])
def test_a_subtypes_own_penalty_decides_before_declaration_order(b, want):
    """Subtypes tie only at the same penalty, and then the one declared first wins."""
    t, y = _operating_day()
    r = optimal_segments(t, y, {"steady": 0}, method="exact", grid_s=5, penalty={"steady": 1.0, "b": b},
                         subtypes={"a": rs.Subtype("steady"), "b": rs.Subtype("steady")})
    assert len(r.segments) > 1 and {s.subtype for s in r.segments} == {want}


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
    want_cost, _ = _brute(t, y, {"steady": 0, "transient": 1}, penalty=0.5, subtypes=subtypes,
                          events=events, **kw)
    got = optimal_segments(t, y, {"steady": 0, "transient": 1}, method="exact", penalty=0.5, subtypes=subtypes,
                           events=events, **kw)
    assert got.objective == pytest.approx(want_cost, rel=1e-9, abs=1e-9)
    for s in got.segments:
        assert _meets_conditions(subtypes[s.subtype], t[s.slice], y[s.slice], _within(events, s.slice))
    for name, cap in caps.items():
        assert (got.counts[name] <= cap).all()


@pytest.mark.parametrize("seed", range(8))
def test_exact_matches_brute_force_with_events_and_constraints_naming_subtypes(seed):
    """Event conditions with transitions, initial and final, durations, caps and penalties that name subtypes."""
    rng = np.random.default_rng(seed)
    n = 7
    t = np.arange(n, dtype=float)
    y = np.where(t < rng.integers(2, 5), 0.0, 3.0) + rng.normal(0, 0.3, n)
    events = {"heater": rng.random(n) < 0.3, "valve": rng.random(n) < 0.3}
    subtypes = {"heat_up": rs.Subtype("transient", events={"heater": True}),
                "drift": rs.Subtype("transient", events={"heater": False}),
                "cold": rs.Subtype("steady", level=(None, 1.5), events={"valve": False}), "hot": rs.Subtype("steady")}
    kw = dict(transitions={"cold": ["heat_up", "drift"], "heat_up": ["hot"], "hot": ["drift", "hot"],
                           "drift": ["cold", "hot"]},
              initial=["cold", "hot"], final=["hot", "drift"], max_counts={"heat_up": 1},
              penalty={"steady": 0.5, "transient": 0.5, "heat_up": 0.2})
    models = {"steady": 0, "transient": 1}
    want_cost, _ = _brute(t, y, models, subtypes=subtypes, events=events, min_d={"drift": 2.0}, **kw)
    got = optimal_segments(t, y, models, method="exact", subtypes=subtypes, events=events,
                           min_duration_s={"drift": 2.0}, **kw)
    assert got.objective == pytest.approx(want_cost, rel=1e-9, abs=1e-9)


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
    assert type(dict(heat_up.events)["heater"]) is bool  # bool, not np.bool_
    stricter = dataclasses.replace(heat_up, max_rise_s=5.0)  # a value derived from another, as for level and peak
    assert stricter.events == heat_up.events and rs.Subtype(**dataclasses.asdict(stricter)) == stricter
    assert rs.Subtype("transient", events=[("heater", True)]) == heat_up  # the pairs it holds
    with pytest.raises(ValueError, match=r"Subtype events must be a mapping \{mask name: True or False\} or "
                                         r"\(mask name, True or False\) pairs"):
        rs.Subtype("transient", events=[("heater", True, 1)])
    for twice in ([("heater", True), ("heater", False)], [("heater", True), ("heater", True)],
                  {("heater", True), ("heater", False)}):  # a set's order would decide which one wins
        with pytest.raises(ValueError, match="Subtype events names mask 'heater' twice"):
            rs.Subtype("transient", events=twice)
    for bad in (["heater"], "heater", True):
        with pytest.raises(ValueError, match=r"Subtype events must be a mapping \{mask name: True or False\}"):
            rs.Subtype("transient", events=bad)
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
    with pytest.raises(ValueError, match=r"events\['heater'\] must be boolean with no missing values, got 2 missing"):
        run({"heater": pd.array([True, None, False, None] + [False] * 6, dtype="boolean")})
    with pytest.raises(ValueError, match=r"events\['valve'\] must be boolean"):  # a mask no condition names
        run({"heater": mask, "valve": mask.astype(int)})
    r = run({"heater": list(mask), "valve": pd.Series(mask)}, subtypes={"heat_up": heat_up,
                                                                       "other": rs.Subtype("transient")})
    assert [s.subtype for s in r.segments] == ["other"] and list(r.info["events"]) == ["heater", "valve"]
    given = np.zeros(10, dtype=bool)
    r = run({"heater": given}, subtypes={"heat_up": heat_up, "other": rs.Subtype("transient")})
    given[3] = True  # the caller reuses its mask: the result keeps the mask it used
    assert not r.info["events"]["heater"].any()

"""
Subtypes of best-fit segments: a heated flow loop over two days, then a
sketch of a reactor's scheme by the same mechanism.

The loop sits at a cold or a hot plateau. A heater switching on starts a
heat-up, and switching it off starts a cool-down. Now and then the loop
drifts with the heater untouched. The facilities, thresholds and caps are
illustrative. Run it with ``python docs/source/examples/subtypes.py``.
"""

from dataclasses import replace

import numpy as np
import pandas as pd

import rocksteady as rs


def flow_loop_record():
    """Two days of loop temperature at one sample a minute, and the heater's switching times."""
    t = pd.date_range("2026-01-05", periods=2 * 24 * 60, freq="1min")
    hours = (t - t[0]) / pd.Timedelta("1h")
    cold, hot = 25.0, 60.0
    y = np.full(len(t), cold)
    for on, off in [(8.0, 16.0), (31.0, 37.0)]:   # day 1 08:00-16:00, day 2 07:00-13:00
        heating = (hours >= on) & (hours < off)
        y[heating] = hot - (hot - cold) * np.exp(-(hours[heating] - on) / 0.25)
        after = hours >= off
        y[after] = cold + (y[heating][-1] - cold) * np.exp(-(hours[after] - off) / 0.33)
    drifting = (hours >= 43.0) & (hours < 45.0)    # day 2 19:00-21:00: the loop drifts up 4 K
    y[drifting] += 2.0 * (hours[drifting] - 43.0)
    y[hours >= 45.0] += 4.0
    y += np.random.default_rng(0).normal(0.0, 0.1, len(t))
    # The heater switches half an hour before the loop temperature answers.
    switched = t.isin(pd.to_datetime(["2026-01-05 07:30", "2026-01-05 15:30",
                                      "2026-01-06 06:30", "2026-01-06 12:30"]))
    return t, y, switched


t, y, switched = flow_loop_record()

# [scheme]
loop = {
    "cold": rs.Subtype("steady", level=(None, 40.0)),          # below 40 °C
    "hot": rs.Subtype("steady", level=(40.0, None)),           # 40 °C and above
    "heat-up": rs.Subtype("transient", events={"heater": True}),
    "cool-down": rs.Subtype("transient", events={"heater": True}),
    "drift": rs.Subtype("transient", events={"heater": False}),
}
transitions = {
    "cold": ["heat-up", "drift"],
    "heat-up": ["hot"],
    "hot": ["cool-down", "drift"],
    "cool-down": ["cold"],
    "drift": ["steady"],                                       # back to either plateau
}
# [scheme]

# [events]
# The log gives the minute the heater switched, and the loop answers about half an hour later.
# A mask has no tolerance, so widen it: flag each switch and the 44 minutes after it.
heater = pd.Series(switched.astype(float), index=t).rolling("45min").max().to_numpy() > 0
# [events]

# [optimise]
res = rs.optimal_segments(
    t, y, {"steady": 0, "transient": "exp"},
    subtypes=loop, events={"heater": heater},
    transitions=transitions,
    initial={"steady"}, final={"steady"},                      # no transient cut off by the record's edge
    per="1D", max_counts={"hot": 1, "heat-up": 1, "cool-down": 1, "drift": 1},
    min_duration_s={"steady": 3600}, max_duration_s={"transient": 3 * 3600},
    method="exact", grid_s=1800,                               # a candidate every 30 minutes
)
print(res)
print(res.counts)
for s in res.segments:
    print(f"{t[s.start]:%a %H:%M} to {t[s.stop - 1]:%a %H:%M}  {s.state.value:9}  {s.subtype}")
# [optimise]

assert [s.subtype for s in res.segments] == [
    "cold", "heat-up", "hot", "cool-down", "cold",             # day 1, and the night into day 2
    "heat-up", "hot", "cool-down", "cold", "drift", "cold"]    # day 2

# [classify]
# The same definitions type segments from anywhere else, such as the detector's.
# On the optimiser's own segments, they give back the optimiser's subtypes.
plain = [replace(s, subtype=None) for s in res.segments]
typed = rs.classify_segments(t, y, plain, subtypes=loop, events={"heater": heater},
                             transitions=transitions, initial={"steady"}, final={"steady"})
assert [s.subtype for s in typed] == [s.subtype for s in res.segments]
# [classify]


# [reactor]
# A reactor's scheme, by the same mechanism. Power is in % of full power, and the thresholds
# are the adapter's to choose; these are illustrative.
P_ZERO, P_OPERATING, P_PULSE, PULSE_RISE_S = 1.0, 10.0, 500.0, 2.0
reactor = {
    "zero-power": rs.Subtype("steady", level=(None, P_ZERO)),
    "at-power": rs.Subtype("steady", level=(P_ZERO, None)),
    "pulse": rs.Subtype("transient", peak=(P_PULSE, None), max_rise_s=PULSE_RISE_S),
    "startup": rs.Subtype("transient", peak=(P_OPERATING, None)),   # must reach operating power
    "shutdown": rs.Subtype("transient"),
    "perturbation": rs.Subtype("transient", events={"perturbing_rod": True}),
    "maneuver": rs.Subtype("transient", events={"control_rods": True}),
    "other": rs.Subtype("transient"),                                # catch-all, declared last
}
reactor_transitions = {
    "zero-power": ["startup", "pulse", "other"],
    "startup": ["at-power"],
    "pulse": ["zero-power"],
    "at-power": ["perturbation", "maneuver", "shutdown", "other"],
    "perturbation": ["at-power"],
    "maneuver": ["at-power"],
    "shutdown": ["zero-power"],
    "other": ["steady"],
}
# [reactor]


def reactor_record():
    """One second per sample: a startup, a maneuver, a perturbation, a shutdown and a pulse, by hand."""
    seconds = np.arange(3300.0)
    power = np.full(len(seconds), 0.01)
    power[300:600] = 50.0 * (1 - np.exp(-(seconds[300:600] - 300) / 60))
    power[600:1200] = 50.0
    power[1200:1300] = 50.0 + 30.0 * (seconds[1200:1300] - 1200) / 100
    power[1300:2400] = 80.0
    power[1800:1830] -= 5.0 * np.sin(np.pi * (seconds[1800:1830] - 1800) / 30)
    power[2400:2700] = 80.0 * np.exp(-(seconds[2400:2700] - 2400) / 40)
    power[3001:3060] = 2000.0 * np.exp(-(seconds[3001:3060] - 3001) / 5)
    cuts = [0, 300, 600, 1200, 1300, 1800, 1830, 2400, 2700, 3000, 3060, 3300]
    states = ["steady", "transient"] * 5 + ["steady"]
    segs = [rs.Segment(a, b, rs.State(st)) for a, b, st in zip(cuts, cuts[1:], states)]
    masks = {"control_rods": np.isin(np.arange(len(seconds)), [300, 1200, 2400]),
             "perturbing_rod": np.isin(np.arange(len(seconds)), [1800])}
    return seconds, power, segs, masks


reactor_t, power, segs, masks = reactor_record()
reactor_typed = rs.classify_segments(reactor_t, power, segs, subtypes=reactor, events=masks,
                                     transitions=reactor_transitions)
assert [s.subtype for s in reactor_typed] == [
    "zero-power", "startup", "at-power", "maneuver", "at-power", "perturbation",
    "at-power", "shutdown", "zero-power", "pulse", "zero-power"]

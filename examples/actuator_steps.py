"""
A process level held by a stepping actuator.

Pattern: discrete actuator moves are found in the data (``StepChange``),
the level must stay within a relative limit of its window mean with a
level-dependent measurement-noise model (``DeviationCriterion``), the
actuator may drift slowly without counting as a move (linear model), and
a shutdown below a floor is steady by definition (``FloorLatch``).

Afterwards the record is split into steady, transient and indeterminate
segments, and each signal is fitted segment by segment: the level with
a constant when steady and a first-order approach when transient, the
actuator with a line. The report gives fit statistics by state and by
segment, with uncertainties corrected for autocorrelation.

Run: python examples/actuator_steps.py [--plot]
"""

import sys

import numpy as np
import pandas as pd

import rocksteady as rs

rng = np.random.default_rng(0)

# --- synthetic record: 1 Hz, two hours --------------------------------------
n, tau = 7200, 60.0  # samples; process time constant in seconds
actuator = np.zeros(n)
actuator[600:] += 20.0  # step up
actuator[2400:] += 10.0  # second step
actuator[4200:] -= 30.0  # shut down
gain, floor = 25.0, 1.0
level = np.zeros(n)
for i in range(1, n):  # first-order response
    level[i] = level[i - 1] + (gain * actuator[i] - level[i - 1]) / tau
level *= 1 + rng.normal(0, 0.004, n)  # 0.4 % multiplicative noise
actuator += rng.normal(0, 0.05, n)
df = pd.DataFrame({"level": level, "actuator": actuator})

# --- detector ---------------------------------------------------------------
signals = [
    rs.Signal("level", lambda s: 0.0 if s["level"] < floor else float(s["level"])),  # zero below the floor
    rs.Signal("actuator", "actuator"),
]


def level_noise(mean_level: float) -> float:
    """Relative standard deviation of a 30 s block mean of the level."""
    return 0.0 if mean_level <= 0 else 0.004 / np.sqrt(30)


criteria = [
    rs.DeviationCriterion("level", limit=0.02, noise=level_noise, model="constant",
                          relative=True, scale_floor=floor, name="level"),
    rs.DeviationCriterion("actuator", limit=1.0, noise=0.05, model="linear", name="actuator"),
]
disturbances = [rs.StepChange("actuator", threshold=2.0, m=10)]
overrides = [rs.FloorLatch("level", floor=floor, release_factor=2.0, name="shutdown")]
timing = rs.Timing(sample_period_s=1.0, block_s=30.0, window_s=300.0, hold_when_undecided=True)
detector = rs.Detector(signals, criteria, disturbances, overrides, timing)

# --- run and report ---------------------------------------------------------
out = rs.batch.run(detector, df)
print("label counts:", out["label"].value_counts().to_dict())
for a, b in rs.batch.accepted_intervals(out["label"]):
    seg = df["level"].iloc[a:b].to_numpy()
    m = rs.stats.mean_with_uncertainty(seg, dt=1.0)
    forced = out["forced_by"].iloc[a:b].dropna().unique().tolist()
    print(f"steady {a:5d}-{b:5d} s  level {m['mean']:8.3f} +/- {m['u_mean']:.3f}"
          + (f"  (forced by {forced[0]})" if forced else ""))

# --- fits and statistics by state -------------------------------------------
report = rs.batch.fit_report(
    detector, df,
    models={"level": {"steady": 0, "transient": "first_order"}, "actuator": 1},
)
print()
print(report)

if "--plot" in sys.argv:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 5))
    ax[0].plot(df["level"], lw=0.6)
    ax[1].plot(df["actuator"], lw=0.6)
    for a, b in rs.batch.accepted_intervals(out["label"]):
        for x in ax:
            x.axvspan(a, b, alpha=0.15, color="tab:green")
    ax[0].set_ylabel("level")
    ax[1].set_ylabel("actuator")
    ax[1].set_xlabel("time (s)")
    plt.show()

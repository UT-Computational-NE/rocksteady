"""
Several temperature channels relaxing towards a setpoint.

Pattern: no actuator is visible in the data; setpoint and configuration
changes come from an operator log (``EventLog``). Steady state means no
resolvable drift (``DriftCriterion``, a slope equivalence test with an
autocorrelation-robust slope uncertainty) on every channel and on derived
signals (mean, spread). Evaluation waits a settling time after each event,
and a passing state must hold for a while before it is accepted.

Run: python examples/thermal_drift.py [--plot]
"""

import sys

import numpy as np
import pandas as pd

import rocksteady as rs
from rocksteady.signals import mean, spread

rng = np.random.default_rng(1)

# --- synthetic record: 1 Hz, four hours -------------------------------------
n, tau = 14400, 600.0  # samples; thermal time constant in seconds
channels = ["T1", "T2", "T3", "T4"]
offsets = np.array([0.0, 0.3, -0.2, 0.6])
setpoint = np.where(np.arange(n) < 7200, 60.0, 70.0)  # setpoint raised at t = 2 h
events = [(7200.0, 0.0)]  # the operator log records the change
core = np.empty(n)
core[0] = 25.0
for i in range(1, n):
    core[i] = core[i - 1] + (setpoint[i] - core[i - 1]) / tau
noise = np.zeros((n, len(channels)))
for i in range(1, n):  # AR(1) sensor noise: correlated from sample to sample
    noise[i] = 0.95 * noise[i - 1] + rng.normal(0, 0.01, len(channels))
df = pd.DataFrame(core[:, None] + offsets + noise, columns=channels)

# --- detector ---------------------------------------------------------------
signals = [rs.Signal(c, c) for c in channels]
signals += [rs.Signal("T_mean", mean(channels)), rs.Signal("T_spread", spread(channels))]
limit = 0.01 / 60.0  # 0.01 K/min, in K/s
criteria = [rs.DriftCriterion(s.name, limit_per_s=limit) for s in signals]
timing = rs.Timing(sample_period_s=1.0, block_s=30.0, window_s=1200.0, settle_s=600.0, hold_s=1200.0)
detector = rs.Detector(signals, criteria, [rs.EventLog(events)], timing=timing)

# --- run and report ---------------------------------------------------------
out = rs.batch.run(detector, df)
print("label counts:", out["label"].value_counts().to_dict())
for a, b in rs.batch.accepted_intervals(out["label"]):
    m = rs.stats.mean_with_uncertainty(df["T1"].iloc[a:b].to_numpy(), dt=1.0)
    print(f"accepted {a:5d}-{b:5d} s  T1 {m['mean']:.3f} +/- {m['u_mean']:.3f} K  (N_eff {m['n_eff']:.0f})")
fails = out[out["label"] == "settling"]["limiting"].value_counts()
print("limiting signal while settling:", fails.head(3).to_dict())

if "--plot" in sys.argv:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 5))
    ax[0].plot(df[channels], lw=0.6)
    ax[1].semilogy(out["eta"], lw=0.6)
    ax[1].axhline(1.0, color="k", lw=0.8)
    for a, b in rs.batch.accepted_intervals(out["label"]):
        for x in ax:
            x.axvspan(a, b, alpha=0.15, color="tab:green")
    ax[0].set_ylabel("temperature (C)")
    ax[1].set_ylabel("margin ratio eta")
    ax[1].set_xlabel("time (s)")
    plt.show()

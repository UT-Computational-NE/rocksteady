"""
Scaling benchmarks for the streaming detector and the shared numerics (#25).

    python tools/benchmark.py            # all
    python tools/benchmark.py slope acf  # some

Sections:

- ``slope``: ``stats.slope_ar1`` against the number of blocks in a window.
- ``detector``: ``Detector.update`` per sample against window length (in
  blocks) and channel count, with a drift and a deviation criterion per
  channel, and the share of time spent copying the window.
- ``acf``: ``stats.acf_tau`` on one long piece plus many short ones
  (fragmented intervals).
- ``events``: ``EventLog.update`` against the number of logged events.

Times are per call, the best of a few repeats. Absolute numbers depend on
the machine; the scaling is what matters.
"""

import sys
import time

import numpy as np

import rocksteady as rs
from rocksteady import stats


def _best(fn, repeat=5, number=None):
    if number is None:  # aim for ~0.05 s per repeat
        number, t = 1, 0.0
        while t < 0.05:
            t0 = time.perf_counter()
            for _ in range(number):
                fn()
            t = time.perf_counter() - t0
            if t < 0.05:
                number *= 4
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(number):
            fn()
        best = min(best, (time.perf_counter() - t0) / number)
    return best


def _us(s):
    return f"{s * 1e6:10.1f} us"


def bench_slope():
    print("slope_ar1 per call, by blocks in the window")
    rng = np.random.default_rng(0)
    for n in (10, 40, 90, 300, 1000, 3000, 10000):
        t = np.arange(n, dtype=float)
        y = rng.normal(size=n).cumsum() * 0.1 + rng.normal(size=n)
        print(f"  n = {n:6d}: {_us(_best(lambda: stats.slope_ar1(t, y), repeat=3))}")


def _detector(n_blocks, channels, block_samples=10):
    names = [f"c{i}" for i in range(channels)]
    sigs = [rs.Signal(c, c) for c in names]
    crit = [rs.DriftCriterion(c, 1.0) for c in names] + [rs.DeviationCriterion(c, 1.0, 0.1) for c in names]
    timing = rs.Timing(block_s=block_samples, window_s=n_blocks * block_samples)
    return rs.Detector(sigs, crit, timing=timing), names


def bench_detector():
    print("Detector.update per sample (steady data: every window evaluated), and the window copy")
    rng = np.random.default_rng(1)
    for n_blocks, channels in ((10, 1), (40, 1), (90, 1), (40, 10), (90, 10), (300, 1), (1000, 1)):
        det, names = _detector(n_blocks, channels)
        n = det.timing.window_samples
        data = [dict(zip(names, row)) for row in rng.normal(0, 0.1, size=(n + 200, channels))]
        for i in range(n):
            det.update(float(i), data[i])
        k = [n]

        def step():
            det.update(float(k[0]), data[n + (k[0] - n) % 200])
            k[0] += 1

        per = _best(step, repeat=3)
        copy = _best(det._window, repeat=3)
        print(f"  {n_blocks:5d} blocks x {channels:2d} channels ({n:6d} samples): {_us(per)}"
              f"   window copy {_us(copy)} ({100 * copy / per:4.1f} %)")


def bench_acf():
    print("acf_tau, pair-weighted: one long piece plus many short pieces")
    rng = np.random.default_rng(2)
    long_piece = rng.normal(size=86400)
    for n_short in (0, 100, 1000, 10000):
        pieces = [long_piece] + [rng.normal(size=5) for _ in range(n_short)]
        print(f"  {n_short:6d} short pieces: {_us(_best(lambda: stats.acf_tau(pieces, 1.0, pair_weighted=True), 3))}")


def bench_events():
    print("EventLog.update per sample, by number of logged events")
    for n_events in (10, 1000, 100000):
        events = [(float(10 * i), 1.0) for i in range(n_events)]
        log = rs.EventLog(events)
        k = [0]

        def step():
            log.update(float(k[0]), {}, {})
            k[0] += 1

        print(f"  {n_events:7d} events: {_us(_best(step, repeat=3))}")


SECTIONS = {"slope": bench_slope, "detector": bench_detector, "acf": bench_acf, "events": bench_events}

if __name__ == "__main__":
    for name in sys.argv[1:] or SECTIONS:
        SECTIONS[name]()
        print()

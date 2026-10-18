"""AsymptoteCriterion: certify that a first-order approach has reached its asymptote (#7)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

import rocksteady as rs
from rocksteady import Outcome

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
from coverage_mc import ar1_noise  # noqa: E402

B = 30.0


def _blocks(n=40):
    t = (np.arange(n) - (n - 1) / 2.0) * B
    return t, t - t[0], (n - 0.5) * B  # centred block times; time from the first block centre to the window end


@pytest.mark.parametrize("tau, outcome", [(200.0, Outcome.PASS), (600.0, Outcome.FAIL)])
def test_settled_passes_and_approaching_fails(tau, outcome):
    t, s, span = _blocks()
    y = 5.0 - 3.0 * np.exp(-s / tau) + np.random.default_rng(0).normal(0, 0.005, len(t))
    r = rs.AsymptoteCriterion("x", 0.05, tau_max=3000.0).evaluate(t, y)
    assert r.outcome == outcome
    true_remaining = -3.0 * np.exp(-span / tau)
    assert r.details["remaining"] == pytest.approx(true_remaining, rel=0.05, abs=2e-3)
    assert r.details["tau_s"] == pytest.approx(tau, rel=0.05)
    assert r.lower <= abs(true_remaining) <= r.upper + 1e-3


def test_flat_window_is_decided_through_tau_max():
    """A flat window has no curve to fit; the bound is then about tau_max times the slope's uncertainty."""
    t, s, span = _blocks()
    y = 5.0 + np.random.default_rng(1).normal(0, 0.005, len(t))
    near = rs.AsymptoteCriterion("x", 0.05, tau_max=2 * span).evaluate(t, y)
    assert near.outcome == Outcome.PASS and near.details["tau_hi"] == pytest.approx(2 * span)
    # With tau_max far beyond the window, a flat window could hide a slow approach: not certified.
    far = rs.AsymptoteCriterion("x", 0.05, tau_max=1000 * span).evaluate(t, y)
    assert far.outcome != Outcome.PASS


def test_slow_approach_is_not_certified():
    """tau longer than the window (within tau_max): the remaining approach is far beyond the limit."""
    t, s, span = _blocks()
    y = 5.0 - 3.0 * np.exp(-s / (3 * span)) + np.random.default_rng(2).normal(0, 0.005, len(t))
    r = rs.AsymptoteCriterion("x", 0.05, tau_max=6 * span).evaluate(t, y)
    assert r.outcome == Outcome.FAIL


def test_relative_mode_and_k():
    t, s, _ = _blocks()
    y = 500.0 - 300.0 * np.exp(-s / 200.0) + np.random.default_rng(3).normal(0, 0.5, len(t))
    absolute = rs.AsymptoteCriterion("x", 5.0, tau_max=3000.0).evaluate(t, y)
    rel = rs.AsymptoteCriterion("x", 0.01, tau_max=3000.0, relative=True).evaluate(t, y)
    assert rel.upper == pytest.approx(absolute.upper / abs(absolute.details["y_inf"]), rel=1e-9)
    zero_k = rs.AsymptoteCriterion("x", 5.0, tau_max=3000.0, k=0.0).evaluate(t, y)
    assert zero_k.lower <= zero_k.upper and np.isfinite(zero_k.ratio)


@pytest.mark.parametrize("kw, match", [
    (dict(limit=1.0, tau_max=0.0), "tau_max must be > 0"),
    (dict(limit=-1.0, tau_max=10.0), "limit must be > 0"),
    (dict(limit=1.0, tau_max=10.0, tau_min=20.0), "tau_min .* <= tau_max"),
    (dict(limit=1.0, tau_max=10.0, confidence=0.4), "confidence must be in"),
    (dict(limit=1.0, tau_max=10.0, n_grid=0), "n_grid must be a positive integer"),
])
def test_validation(kw, match):
    with pytest.raises(ValueError, match=match):
        rs.AsymptoteCriterion("x", **kw)


def test_spec_detector_and_diagnostic_columns():
    spec = {"timing": {"block_s": 10, "window_s": 300}, "signals": ["x"],
            "criteria": [{"type": "asymptote", "signal": "x", "limit": 0.05, "tau_max": 600}]}
    det = rs.build_detector(spec)
    t = np.arange(1500.0)
    x = 20.0 - 10.0 * np.exp(-t / 120.0) + np.random.default_rng(4).normal(0, 0.01, len(t))
    out = rs.batch.run(det, pd.DataFrame({"x": x}), with_results=True)
    cols = [f"detail_x_asymptote_{k}" for k in rs.AsymptoteCriterion.detail_keys]
    assert set(cols) <= set(out.columns) and out[cols].dtypes.eq(float).all()
    assert out["label"].iloc[-1] == "steady"
    assert out["detail_x_asymptote_tau_s"].dropna().iloc[-1] > 0
    # The same schema for an empty record.
    empty = rs.batch.run(det, pd.DataFrame({"x": []}), with_results=True)
    assert list(empty.columns) == list(out.columns)
    # Fewer than 10 blocks per window is refused.
    with pytest.raises(ValueError, match="at least 10 blocks"):
        rs.build_detector({**spec, "timing": {"block_s": 10, "window_s": 90}})


def test_certifies_a_slow_correlated_approach_the_drift_test_cannot():
    """The motivation: strongly autocorrelated block means about a line during a thermal approach."""
    rng = np.random.default_rng(5)
    t = np.arange(0.0, 4 * 3600.0, 10.0)  # 10 s samples, 3 per block
    noise = np.repeat(ar1_noise(rng, len(t) // 3, 0.6), 3) * 0.004
    x = 40.0 - 4.0 * np.exp(-t / 1500.0) + noise
    timing = rs.Timing(sample_period_s=10, block_s=30, window_s=3600)
    drift = rs.Detector([rs.Signal("x", "x")], [rs.DriftCriterion("x", 1e-5)], timing=timing)
    asym = rs.Detector([rs.Signal("x", "x")], [rs.AsymptoteCriterion("x", 0.05, tau_max=3000.0)], timing=timing)
    df = pd.DataFrame({"x": x})
    first = {}
    for name, det in (("drift", drift), ("asymptote", asym)):
        labels = rs.batch.run(det, df, t=t)["label"].to_numpy()
        hit = np.flatnonzero(labels == "steady")
        first[name] = float(t[hit[0]]) if len(hit) else None
    # Remaining approach 4 exp(-t/1500) falls below 0.05 K at about 6570 s; the drift limit 1e-5 K/s
    # (4/1500 exp(-t/1500) < 1e-5) is reached only at about 8380 s, so drift certifies much later.
    assert first["asymptote"] is not None and first["asymptote"] < 8000
    assert first["drift"] is None or first["drift"] > first["asymptote"]


@pytest.mark.slow
@pytest.mark.parametrize("phi", [0.0, 0.5, 0.8])
@pytest.mark.parametrize("ratio", [0.5, 1.0, 2.0])
def test_false_pass_rate_at_the_limit(phi, ratio):
    """True remaining approach equal to the limit, tau_max = 2 tau, 40 blocks: nominal 2.5 %."""
    rng = np.random.default_rng(int(10 * phi + ratio * 3))
    t, s, span = _blocks(40)
    tau = span / ratio
    amp = np.exp(span / tau)  # remaining approach 1 at the window end
    crit = rs.AsymptoteCriterion("x", 1.0, tau_max=2 * tau)
    reps = 600
    passes = sum(crit.evaluate(t, -amp * np.exp(-s / tau) + 0.1 * np.sqrt(1 - phi * phi) * ar1_noise(rng, 40, phi))
                 .outcome == Outcome.PASS for _ in range(reps))
    assert passes / reps <= 0.05  # at most twice nominal (binomial sd about 0.6 % at 2.5 %)

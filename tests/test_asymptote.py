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
    # Centred block times; time from the first block centre; and the evaluation time of a criterion used on
    # its own (no bound timing): the last block centre.
    return t, t - t[0], (n - 1) * B


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
    t, _, span = _blocks()
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
    from coverage_mc import asymptote_pass_rate

    rate = asymptote_pass_rate(40, phi, ratio, reps=600, seed=int(10 * phi + 3 * ratio))
    assert rate <= 0.05  # at most twice nominal (binomial sd about 0.6 % at 2.5 %)


@pytest.mark.slow
@pytest.mark.parametrize("ratio", [0.5, 1.0, 2.0])
def test_false_pass_rate_with_partial_blocks(ratio):
    """Random tolerated gaps (about 30 % of samples, coverage >= 0.5 per block, current sample valid)."""
    rng = np.random.default_rng(int(7 * ratio))
    n, b = 40, 30
    timing = rs.Timing(sample_period_s=1.0, block_s=b, window_s=n * b, max_gap_s=b, min_block_coverage=0.5)
    t = (np.arange(n) - (n - 1) / 2.0) * b
    samples = (t - 0.5 * (b - 1))[:, None] + np.arange(b)[None, :]
    s_end = (t[-1] + 0.5 * (b - 1)) - samples[0, 0]
    tau = s_end / ratio
    amp = np.exp(s_end / tau)  # remaining approach 1 (the limit) at the current sample
    curve = -amp * np.exp(-(samples - samples[0, 0]) / tau)
    crit = rs.AsymptoteCriterion("x", 1.0, tau_max=2 * tau)
    crit.bind_timing(timing)
    reps, passes = 400, 0
    for _ in range(reps):
        valid = rng.random((n, b)) > 0.3
        valid[valid.mean(axis=1) < 0.5] = True  # every block meets coverage 0.5
        valid[-1, -1] = True                    # the current sample is valid
        means = np.where(valid, curve, 0.0).sum(axis=1) / valid.sum(axis=1)
        noisy = means + 0.1 * rng.normal(size=n) / np.sqrt(valid.sum(axis=1) / b)
        passes += crit.evaluate(t, noisy, valid=valid).outcome == Outcome.PASS
    assert passes / reps <= 0.05


# -- the current sample, block timing and input checks --------------------------------


def test_remaining_approach_is_taken_at_the_current_sample():
    """One-sample blocks: the last block centre is the current sample; nothing may be extrapolated."""
    det = rs.Detector([rs.Signal("x", "x")], [rs.AsymptoteCriterion("x", 1.0, tau_max=60)],
                      timing=rs.Timing(sample_period_s=30, block_s=30, window_s=300))
    t = np.arange(10) * 30.0
    x = 5 - 1.3 * np.exp((270 - t) / 30)  # 1.3 left at the last sample: above the limit of 1
    v = [det.update(float(ti), {"x": xi}) for ti, xi in zip(t, x)][-1]
    assert v.label != rs.Label.STEADY
    assert v.results["x_asymptote"].details["remaining"] == pytest.approx(-1.3, rel=1e-6)


@pytest.mark.parametrize("block_samples", [1, 3, 10])
def test_offset_to_the_current_sample_follows_the_block_length(block_samples):
    dt = 2.0
    timing = rs.Timing(sample_period_s=dt, block_s=block_samples * dt, window_s=40 * block_samples * dt)
    crit = rs.AsymptoteCriterion("x", 1.0, tau_max=1e4)
    det = rs.Detector([rs.Signal("x", "x")], [crit], timing=timing)
    t = np.arange(40 * block_samples) * dt
    tau = 0.3 * t[-1]
    x = 10 - 4 * np.exp(-t / tau)
    v = [det.update(float(ti), {"x": xi}) for ti, xi in zip(t, x)][-1]
    # Noiseless data: the basis is averaged over each block's samples, as the block means are, so the
    # remaining approach at the last sample is exact.
    got, want = v.results["x_asymptote"].details["remaining"], -4 * np.exp(-t[-1] / tau)
    assert got == pytest.approx(want, rel=1e-6)
    with pytest.raises(ValueError, match="already used with another timing"):
        other = rs.Timing(sample_period_s=dt, block_s=(block_samples + 1) * dt, window_s=40 * (block_samples + 1) * dt)
        rs.Detector([rs.Signal("x", "x")], [crit], timing=other)


def test_grid_must_hold_both_bounds():
    with pytest.raises(ValueError, match="n_grid must be at least 2"):
        rs.AsymptoteCriterion("x", 0.05, tau_max=7110, n_grid=1)
    t = np.arange(40) * 30.0
    y = 5 - 3 * np.exp(-t / 3555)
    r = rs.AsymptoteCriterion("x", 0.05, tau_max=7110, n_grid=2).evaluate(t, y)
    assert r.outcome != Outcome.PASS and r.details["tau_hi"] == pytest.approx(7110)


def test_relative_mode_with_a_zero_level_is_undecided():
    for crit in (rs.AsymptoteCriterion("x", 0.05, tau_max=3000, relative=True),
                 rs.DeviationCriterion("x", 0.05, 0.0, relative=True)):
        r = crit.evaluate(np.arange(40) * 30.0, np.zeros(40))
        assert r.outcome == Outcome.UNDECIDED and r.invalid_reason is not None
    det = rs.Detector([rs.Signal("x", "x")], [rs.AsymptoteCriterion("x", 0.05, tau_max=3000, relative=True)],
                      timing=rs.Timing(block_s=30, window_s=1200))
    out = rs.batch.run(det, pd.DataFrame({"x": np.zeros(1300)}))
    assert out["reason"].iloc[-1] == "invalid_result"  # a controlled result, not an exception
    floor = rs.AsymptoteCriterion("x", 0.05, tau_max=3000, relative=True, scale_floor=1.0)
    assert floor.evaluate(np.arange(40) * 30.0, np.zeros(40)).outcome == Outcome.PASS


def test_colliding_diagnostic_columns_are_rejected_before_reading_input():
    class Diag(rs.Criterion):
        min_blocks = 3

        def __init__(self, name, key):
            self.name, self.signal, self.detail_keys = name, "x", (key,)

        def evaluate(self, t_blocks, y_blocks):
            return rs.CriterionResult(self.name, 0.0, 0.0, 1.0, 1.0, {self.detail_keys[0]: 1.0})

    det = rs.Detector([rs.Signal("x", "x")], [Diag("a", "b_c"), Diag("a_b", "c")],
                      timing=rs.Timing(block_s=1, window_s=3))
    for df in (pd.DataFrame({"x": np.zeros(4)}), pd.DataFrame({"x": []})):
        with pytest.raises(ValueError, match="'detail_a_b_c' comes from both criterion 'a' key 'b_c' and criterion"):
            rs.batch.run(det, df, with_results=True)
    assert len(rs.batch.run(det, pd.DataFrame({"x": np.zeros(4)}))) == 4  # without results, no diagnostic columns


@pytest.mark.parametrize("noise_sd, seed", [(0.0, 0)] + [(0.001, s) for s in range(5)])
def test_partial_blocks_are_not_extrapolated(noise_sd, seed):
    """Tolerated gaps: each two-sample block is [NaN, valid], so its mean was measured at its last sample."""
    t = np.arange(22) * 30.0
    x = 5 - 1.15 * np.exp((t[-1] - t) / 60.0) + np.random.default_rng(seed).normal(0, noise_sd, len(t))
    x[::2] = np.nan
    timing = rs.Timing(sample_period_s=30, block_s=60, window_s=600, max_gap_s=30, min_block_coverage=0.5)
    det = rs.Detector([rs.Signal("x", "x")], [rs.AsymptoteCriterion("x", 1.0, tau_max=120)], timing=timing)
    v = [det.update(float(ti), {"x": float(xi)}) for ti, xi in zip(t, x)][-1]
    assert v.label != rs.Label.STEADY  # 1.15 left at the last sample, above the limit of 1
    if noise_sd == 0:
        assert v.results["x_asymptote"].details["remaining"] == pytest.approx(-1.15, rel=1e-6)


def test_criteria_receive_the_times_their_block_means_were_measured_at():
    seen = []

    class Spy(rs.Criterion):
        min_blocks = 3

        def __init__(self):
            self.name, self.signal = "spy", "x"

        def evaluate(self, t_blocks, y_blocks):
            seen.append(np.array(t_blocks))
            return rs.CriterionResult("spy", 0.0, 0.0, 1.0, 1.0)

    timing = rs.Timing(sample_period_s=1, block_s=4, window_s=12, max_gap_s=3, min_block_coverage=0.25)
    det = rs.Detector([rs.Signal("x", "x")], [Spy()], timing=timing)
    for i in range(12):
        det.update(float(i), {"x": 0.0})
    nominal = seen[-1].copy()
    np.testing.assert_array_equal(nominal, [-4.0, 0.0, 4.0])  # complete blocks: the block centres
    # The next window, samples 12-23, after valid history: block 1 keeps only its last sample (offset +1.5);
    # block 3 loses its third sample, so its valid offsets are -1.5, -0.5, +1.5 (mean -1/6).
    for i in range(12, 24):
        det.update(float(i), {"x": float("nan") if i in (12, 13, 14, 22) else 0.0})
    np.testing.assert_allclose(seen[-1], [-4.0 + 1.5, 0.0, 4.0 - 1.0 / 6.0])



@pytest.mark.parametrize("noise_sd", [0.0, 1e-7, 1e-6])
def test_block_masks_that_vary_between_blocks(noise_sd):
    """A centroid time is exact for a line, not for an exponential: the basis must average the valid samples."""
    t = np.arange(40, dtype=float)
    tau = 39 / 8
    x = 5 - 0.00103 * np.exp((39 - t) / tau) + np.random.default_rng(1).normal(0, noise_sd, 40)
    masks = ["1001", "1100", "0111", "0110", "1010", "0110", "0111", "1001", "0111", "1111"]
    x[~np.array([c == "1" for c in "".join(masks)])] = np.nan
    timing = rs.Timing(sample_period_s=1, block_s=4, window_s=40, max_gap_s=3, min_block_coverage=0.5)
    det = rs.Detector([rs.Signal("x", "x")], [rs.AsymptoteCriterion("x", 0.001, tau_max=2 * tau)], timing=timing)
    v = [det.update(float(ti), {"x": float(xi)}) for ti, xi in zip(t, x)][-1]
    r = v.results["x_asymptote"]
    assert v.label != rs.Label.STEADY and r.upper > 0.001  # 0.00103 left, above the limit of 0.001
    if noise_sd == 0:
        assert r.details["remaining"] == pytest.approx(-0.00103, rel=1e-6)


def test_mask_shape_is_checked():
    timing = rs.Timing(sample_period_s=1, block_s=4, window_s=40)
    crit = rs.AsymptoteCriterion("x", 0.1, tau_max=100.0)
    crit.bind_timing(timing)
    t = (np.arange(10) - 4.5) * 4.0
    with pytest.raises(ValueError, match="valid has shape"):
        crit.evaluate(t, np.zeros(10), valid=np.ones((10, 3), dtype=bool))
    with pytest.raises(ValueError, match="9 blocks, but its detector has 10"):
        crit.evaluate(t[:9], np.zeros(9))

"""Required block and window lengths, the calibration warning, and suggest_timing (#51)."""

import os
import sys
import warnings

import numpy as np
import pytest

import rocksteady as rs
from rocksteady import CalibrationWarning, stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
from coverage_mc import ar1_noise  # noqa: E402


def test_block_and_window_are_required():
    with pytest.raises(ValueError, match="Timing needs block_s and window_s: .* rs.suggest_timing"):
        rs.Timing()
    with pytest.raises(ValueError, match="Timing needs window_s"):
        rs.Timing(block_s=30.0)
    assert rs.Timing(1.0, 30.0, 1200.0).n_blocks == 40  # positional use still works
    with pytest.raises(ValueError, match="Detector needs a timing"):
        rs.Detector([rs.Signal("x", "x")], [rs.DeviationCriterion("x", 1.0, 0.1)])


@pytest.mark.parametrize("make, n_blocks, warns", [
    (lambda: rs.DriftCriterion("x", 1e-3), 10, True),
    (lambda: rs.DriftCriterion("x", 1e-3), 40, False),
    (lambda: rs.BoundCriterion("x", 0.0, 1.0), 5, True),
    (lambda: rs.BoundCriterion("x", 0.0, 1.0), 10, False),
    (lambda: rs.AsymptoteCriterion("x", 0.1, tau_max=100.0), 15, True),
    (lambda: rs.DeviationCriterion("x", 1.0, 0.1), 3, False),  # calibrated with its noise model, no minimum
])
def test_too_few_blocks_warns(make, n_blocks, warns):
    timing = rs.Timing(block_s=10.0, window_s=10.0 * n_blocks)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rs.Detector([rs.Signal("x", "x")], [make()], timing=timing)
    hits = [w for w in caught if issubclass(w.category, CalibrationWarning)]
    assert bool(hits) == warns
    if warns:
        assert f"holds {n_blocks} blocks" in str(hits[0].message) and "suggest_timing" in str(hits[0].message)


def _ar1_samples(rng, n, phi, sd):
    return sd * np.sqrt(1 - phi * phi) * ar1_noise(rng, n, phi)


def test_suggest_timing_measures_both_time_scales():
    rng = np.random.default_rng(0)
    phi = 0.9  # tau_int = (1 + phi) / (1 - phi) = 19 s at 1 Hz
    steady = 20 + _ar1_samples(rng, 20000, phi, 0.05)
    t = np.arange(4000.0)
    transient = 20 + 5 * (1 - np.exp(-t / 600.0)) + _ar1_samples(rng, 4000, phi, 0.05)
    s = rs.suggest_timing(steady, 1.0, transient=transient)
    assert s.noise_tau_s == pytest.approx(19.0, rel=0.3)
    assert s.block_s == round(3 * s.noise_tau_s) and s.n_blocks == 40 and s.window_s == 40 * s.block_s
    assert s.process_tau_s == pytest.approx(600.0, rel=0.1)
    assert s.block_mean_phi < 0.4
    timing = s.timing(settle_s=60.0)
    assert (timing.block_s, timing.window_s, timing.settle_s) == (s.block_s, s.window_s, 60.0)
    assert "block_s =" in str(s) and any("nearly independent" in note for note in s.notes)


def test_suggest_timing_reports_conflicts_and_limits():
    rng = np.random.default_rng(1)
    steady = _ar1_samples(rng, 20000, 0.9, 0.05)
    t = np.arange(1000.0)
    fast = rs.suggest_timing(steady, 1.0, transient=5 * (1 - np.exp(-t / 100.0)), max_window_s=500.0)
    assert fast.block_s == 20.0  # capped at 0.2 x tau = 20 s, below the 3 x 19 s the noise asks for
    assert any(note.startswith("CONFLICT") for note in fast.notes)
    assert fast.n_blocks == 25 and any("holds only 25 blocks" in note for note in fast.notes)
    with pytest.raises(ValueError, match="at least 20 finite samples"):
        rs.suggest_timing(np.ones(10), 1.0)
    with pytest.raises(ValueError, match="holds 1 block.* needs at least 3"):
        rs.suggest_timing(np.random.default_rng(0).normal(size=10000), 1.0, max_window_s=5.0)
    with pytest.raises(ValueError, match="dt must be a positive"):
        rs.suggest_timing(steady, 0.0)
    unresolved = rs.suggest_timing(steady, 1.0, transient=np.zeros(200))
    assert np.isnan(unresolved.process_tau_s) and any("not resolved" in n for n in unresolved.notes)


@pytest.mark.slow
def test_a_suggested_timing_keeps_the_drift_test_calibrated():
    """Sample-level AR(1) noise with a long correlation time; slope at the limit: about nominal 2.5 %."""
    rng = np.random.default_rng(2)
    phi, sd = 0.95, 0.05  # tau_int = 39 s
    s = rs.suggest_timing(_ar1_samples(rng, 40000, phi, sd), 1.0)
    timing = s.timing()
    b, n = timing.block_samples, timing.n_blocks
    t_blocks = (np.arange(n) - (n - 1) / 2.0) * timing.effective_block_s
    limit = 5 * sd / timing.window_s  # a slope comparable to the noise over the window
    reps, passes = 400, 0
    for _ in range(reps):
        x = _ar1_samples(rng, n * b, phi, sd) + limit * np.arange(n * b)
        bv, ub, k = stats.slope_ar1(t_blocks, x.reshape(n, b).mean(axis=1))
        passes += abs(bv) + k * ub <= limit
    assert passes / reps <= 0.05


def test_missing_samples_keep_their_place_on_the_time_grid():
    """Correlation is measured within contiguous runs, never across a gap (the time axis is not compressed)."""
    x = ar1_noise(np.random.default_rng(2026), 100000, 0.9)  # tau_int = 19 s at 1 Hz
    full = rs.suggest_timing(x, 1.0)
    runs = x.copy()
    runs[(np.arange(len(x)) % 1000) < 50] = np.nan  # 50-sample gaps at known positions
    gapped = rs.suggest_timing(runs, 1.0)
    assert gapped.noise_tau_s == pytest.approx(full.noise_tau_s, rel=0.05)
    assert abs(gapped.block_s - full.block_s) <= 2 and gapped.block_mean_phi < 0.25
    assert any("never across a gap" in note for note in gapped.notes)
    # Nine in ten missing: no run to measure a correlation in. Compressing the gaps would have given a
    # correlation time ten times too short and blocks that look independent but are strongly correlated.
    sparse = x.copy()
    sparse[np.arange(len(x)) % 10 != 0] = np.nan
    with pytest.raises(ValueError, match="no run of at least 20 consecutive valid samples"):
        rs.suggest_timing(sparse, 1.0)


@pytest.mark.parametrize("cap", [9.0, 10.0, 50.0, 119.0, 1e6])
def test_the_window_never_exceeds_max_window_s(cap):
    s = rs.suggest_timing(np.random.default_rng(3).normal(size=10000), 1.0, max_window_s=cap)
    assert 3 <= s.n_blocks <= 40 and s.window_s <= cap


@pytest.mark.parametrize("level", [0.0, 5.0])
def test_constant_steady_data_give_an_undefined_correlation_quietly(level):
    with np.errstate(all="raise"), warnings.catch_warnings():
        warnings.simplefilter("error")
        s = rs.suggest_timing(np.full(1000, level), 1.0)
    assert np.isnan(s.block_mean_phi) and any("constant" in note for note in s.notes)
    assert s.block_s >= 1.0 and s.window_s <= 40 * s.block_s

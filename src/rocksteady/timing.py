"""
Choosing block and window lengths from data.

Block and window lengths depend on two time scales of the application's own
data, so rocksteady has no default for them:

- **block length**: several times the *noise correlation time* of the raw
  samples, so block means are nearly independent, but short compared with
  the *process time scale* that should still be resolved;
- **window**: enough blocks for the criteria's statistics (about 40 for the
  ``"ar1"`` drift test), and no longer than the reaction time that can be
  accepted, since every steady call lags by about one window.

``suggest_timing`` measures both time scales: the noise correlation time on
a stretch known to be steady, and optionally the time constant of a typical
transient. It returns a proposal with its reasoning, which the adapter
reviews and adopts; nothing is set automatically.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .core import Timing
from .stats import acf_tau, linear_fit


@dataclass
class TimingSuggestion:
    """
    A proposed block and window length, with the measurements behind it.

    ``noise_tau_s`` is the integrated autocorrelation time of the steady
    stretch (about a line); ``process_tau_s`` the time constant of the
    transient (NaN if none was given or it could not be resolved);
    ``block_mean_phi`` the lag-1 autocorrelation of block means at the
    suggested block length, measured on the steady stretch (NaN when it is
    too short to tell). ``notes`` explain each choice and any conflict.
    """

    sample_period_s: float
    block_s: float
    window_s: float
    n_blocks: int
    noise_tau_s: float
    process_tau_s: float = float("nan")
    block_mean_phi: float = float("nan")
    notes: List[str] = field(default_factory=list)

    def timing(self, **overrides) -> Timing:
        """A ``Timing`` with the suggested lengths; other options (``settle_s``, ``hold_s``, ...) as keywords."""
        kw = {"sample_period_s": self.sample_period_s, "block_s": self.block_s, "window_s": self.window_s}
        kw.update(overrides)
        return Timing(**kw)

    def __str__(self) -> str:
        head = (f"block_s = {self.block_s:g} s, window_s = {self.window_s:g} s ({self.n_blocks} blocks); "
                f"noise correlation time {self.noise_tau_s:.3g} s")
        if math.isfinite(self.process_tau_s):
            head += f", process time constant {self.process_tau_s:.3g} s"
        return "\n".join([head] + [f"- {n}" for n in self.notes])


def _lag1(x: np.ndarray) -> float:
    x = x - x.mean()
    xx = float(x @ x)
    return float(x[1:] @ x[:-1]) / xx if xx > 0 and len(x) > 2 else float("nan")


def suggest_timing(steady, dt: float, *, transient=None, transient_t=None, block_factor: float = 3.0,
                   min_blocks: int = 40, max_window_s: Optional[float] = None,
                   block_to_process: float = 0.2) -> TimingSuggestion:
    """
    Propose ``block_s`` and ``window_s`` from the application's data.

    Parameters
    ----------
    steady : array
        A stretch of one signal known to be steady, sampled every ``dt``
        seconds. Non-finite samples are missing samples: they keep their
        place on the time grid, and correlations are measured within the
        contiguous runs of valid samples, never across a gap. At least one
        run of 20 valid samples is needed. Longer is better: a few hundred
        noise correlation times.
    dt : float
        Sample period in seconds.
    transient : array, optional
        A typical transient of the same signal (a step response), for its
        time constant; ``transient_t`` gives its times (default: every ``dt``).
    block_factor : float
        Block length in noise correlation times (default 3: block means then
        have a lag-1 autocorrelation of roughly 0.1 to 0.3).
    min_blocks : int
        Blocks per window (default 40, where the ``"ar1"`` drift test holds
        its calibration).
    max_window_s : float, optional
        The longest window the application accepts (its reaction time). The
        returned window never exceeds it. If ``min_blocks`` blocks do not
        fit, fewer are used and a note says so; if not even 3 fit, a
        ``ValueError`` explains the choices (a longer window, or shorter,
        correlated blocks).
    block_to_process : float
        Largest block length as a fraction of the process time constant
        (default 0.2), so blocks follow the transient.

    Returns
    -------
    TimingSuggestion
        Use ``suggestion.timing(settle_s=..., hold_s=...)`` for a ``Timing``,
        once reviewed.
    """
    dt = float(dt)
    if not dt > 0 or not math.isfinite(dt):
        raise ValueError(f"dt must be a positive number of seconds, got {dt!r}")
    if not block_factor > 0 or not 0 < block_to_process <= 1 or int(min_blocks) < 3:
        raise ValueError("block_factor must be > 0, block_to_process in (0, 1], min_blocks >= 3")
    y = np.asarray(steady, dtype=float)
    ok = np.isfinite(y)
    if int(ok.sum()) < 20:
        raise ValueError(f"steady needs at least 20 finite samples, got {int(ok.sum())}")
    # Missing samples keep their place on the time grid: samples either side of a gap are not adjacent
    # observations. The line is fitted at the samples' true times, and lag products are pooled within
    # contiguous runs of valid samples only (stats.acf_tau).
    pos = np.arange(len(y), dtype=float)
    _, _, r = linear_fit(pos[ok], y[ok])
    resid = np.full(len(y), np.nan)
    resid[ok] = r
    edges = np.flatnonzero(np.diff(np.concatenate(([0], ok.astype(np.int8), [0]))))
    pieces = [resid[a:b] for a, b in zip(edges[::2], edges[1::2])]
    longest = max(len(p) for p in pieces)
    if longest < 20:
        raise ValueError(f"steady has no run of at least 20 consecutive valid samples (the longest is {longest}); "
                         "the correlation time cannot be measured across gaps. Give a gap-free steady stretch.")
    notes: List[str] = []
    noise_tau = float(acf_tau(pieces, dt))
    if not ok.all():
        notes.append(f"{int((~ok).sum())} missing samples: correlation measured within {len(pieces)} contiguous "
                     f"runs (longest {longest} samples), never across a gap")
    if ok.sum() * dt < 100 * noise_tau:
        notes.append(f"the valid samples span only {ok.sum() * dt / noise_tau:.0f} noise correlation times; the "
                     "estimate is rough (a few hundred are better)")
    samples = max(1, int(round(block_factor * noise_tau / dt)))
    notes.append(f"block: {block_factor:g} x the noise correlation time {noise_tau:.3g} s, rounded to "
                 f"{samples} sample{'s' if samples != 1 else ''}, so block means are nearly independent")

    process_tau = float("nan")
    if transient is not None:
        from .fit import fit_first_order

        yt = np.asarray(transient, dtype=float)
        tt = np.arange(len(yt)) * dt if transient_t is None else np.asarray(transient_t, dtype=float)
        r = fit_first_order(tt, yt)
        if r.flags & {"tau_unidentifiable", "tau_at_bound", "too_few_samples"}:
            notes.append(f"the transient's time constant is not resolved ({', '.join(sorted(r.flags))}); "
                         "it was not used")
        else:
            process_tau = float(r.params["tau"])
            cap = max(1, int(math.floor(block_to_process * process_tau / dt + 1e-9)))  # rounding of a fitted tau
            if samples > cap:
                notes.append(f"CONFLICT: the noise wants blocks of {samples * dt:g} s, but the process time "
                             f"constant {process_tau:.3g} s allows at most {cap * dt:g} s "
                             f"({block_to_process:g} tau); using {cap * dt:g} s, so block means stay correlated "
                             "and the drift test's k grows. Consider a less noisy signal or averaging channels.")
                samples = cap
            else:
                notes.append(f"the block is at most {block_to_process:g} x the process time constant "
                             f"{process_tau:.3g} s, so blocks follow the transient")
    block_s = samples * dt

    n_blocks = int(min_blocks)
    if max_window_s is not None:
        fit_blocks = int(math.floor(float(max_window_s) / block_s + 1e-9))
        if fit_blocks < 3:
            raise ValueError(
                f"max_window_s = {max_window_s:g} s holds {fit_blocks} block(s) of {block_s:g} s, and a detector "
                f"needs at least 3 (the drift test 4, and about 20-40 for its calibration). Accept a longer "
                f"window (at least {3 * block_s:g} s), or shorter blocks (a smaller block_factor), knowing that "
                "block means then stay correlated.")
        if fit_blocks < n_blocks:
            notes.append(f"max_window_s = {max_window_s:g} s holds only {fit_blocks} blocks, fewer than "
                         f"{min_blocks}; the drift test's calibration weakens below about 40 blocks and fails "
                         "below about 20 (see the calibration tables)")
            n_blocks = fit_blocks
    window_s = n_blocks * block_s
    notes.append(f"window: {n_blocks} blocks = {window_s:g} s; a steady call lags by about one window")
    if math.isfinite(process_tau) and window_s < process_tau:
        notes.append(f"the window is shorter than the process time constant {process_tau:.3g} s: an asymptote "
                     "test (AsymptoteCriterion) has little power on approaches that slow, and a drift test sees "
                     "a nearly linear approach")

    # Block means on the original grid; only complete blocks, and only pairs of consecutive complete blocks.
    nb = len(y) // samples
    phi_b = float("nan")
    if nb >= 2:
        grid = resid[: nb * samples].reshape(nb, samples)
        full = np.isfinite(grid).all(axis=1)
        means = np.full(nb, np.nan)
        means[full] = grid[full].mean(axis=1)
        pair = full[1:] & full[:-1]
        if int(pair.sum()) >= 10:
            m = means - np.nanmean(means)
            phi_b = float((m[1:][pair] * m[:-1][pair]).mean() / np.nanmean(m[full] ** 2))
            notes.append(f"measured lag-1 autocorrelation of block means on the steady stretch: {phi_b:.2f} "
                         f"({int(pair.sum())} pairs of consecutive complete blocks)")
    return TimingSuggestion(dt, block_s, window_s, n_blocks, noise_tau, process_tau, phi_b, notes)


__all__ = ["TimingSuggestion", "suggest_timing"]
